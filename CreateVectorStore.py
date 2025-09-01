import os
from dotenv import load_dotenv
# from langchain_community.document_loaders import PyMuPDFLoader   # ไม่ใช้แล้ว (PDF)
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import re

# IFC
import ifcopenshell  # IFC
from glob import glob  # IFC

VECTOR_STORE_PATH = "Data/vector_store"

def clean_text(text):
    text = text.replace('\uf70b', '้')
    text = text.replace('\uf70a', '่')
    text = text.replace('\uf701', '')
    text = text.replace('\uf712', '็')
    text = text.replace('\uf70e', '์')
    text = text.replace('\uf710', 'ั')
    text = text.replace('\uf702', 'ำ')
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# ---------- IFC: ฟังก์ชันแปลง IFC -> ข้อความ ----------
def extract_ifc_text(path):
    try:
        ifc = ifcopenshell.open(path)
    except Exception as e:
        return f"ไม่สามารถเปิดไฟล์ IFC: {path} เหตุผล: {e}"

    parts = []

    # Project / Site / Building / Storey (ภาพรวมโครงสร้างเชิงลำดับชั้น)
    def safe(v):
        try:
            return str(v) if v is not None else ""
        except Exception:
            return ""

    proj = ifc.by_type("IfcProject")
    if proj:
        p = proj[0]
        parts.append(f"PROJECT: Name={safe(getattr(p,'Name',None))} GlobalId={safe(getattr(p,'GlobalId',None))}")

    for cls in ["IfcSite", "IfcBuilding", "IfcBuildingStorey"]:
        for ent in ifc.by_type(cls):
            name = safe(getattr(ent, "Name", None))
            gid = safe(getattr(ent, "GlobalId", None))
            elev = safe(getattr(ent, "Elevation", None)) if hasattr(ent, "Elevation") else ""
            extra = f" Elevation={elev}" if elev != "" else ""
            parts.append(f"{cls}: Name={name} GlobalId={gid}{extra}")

    # องค์ประกอบหลักที่ถามบ่อย (นับ/ค้นหาได้ดี)
    product_types = [
        "IfcWall", "IfcSlab", "IfcColumn", "IfcBeam",
        "IfcDoor", "IfcWindow", "IfcSpace",
        "IfcStair", "IfcRailing", "IfcRoof"
    ]

    for t in product_types:
        elements = ifc.by_type(t)
        for e in elements:
            name = safe(getattr(e, "Name", None))
            gid = safe(getattr(e, "GlobalId", None))
            tag = safe(getattr(e, "Tag", None))
            line = [f"{t}", f"Name={name}", f"GlobalId={gid}"]
            if tag:
                line.append(f"Tag={tag}")

            # ดึง PropertySet แบบย่อ (จำกัดจำนวนเพื่อไม่ให้ยาวเกิน)
            pset_items = []
            try:
                rels = getattr(e, "IsDefinedBy", []) or []
                for rel in rels:
                    if rel.is_a("IfcRelDefinesByProperties"):
                        pdef = rel.RelatingPropertyDefinition
                        if pdef and pdef.is_a("IfcPropertySet"):
                            pset_name = safe(getattr(pdef, "Name", None))
                            for prop in pdef.HasProperties or []:
                                if prop.is_a("IfcPropertySingleValue"):
                                    pname = safe(getattr(prop, "Name", None))
                                    nv = getattr(prop, "NominalValue", None)
                                    val = ""
                                    if nv is not None:
                                        # บางครั้งเป็น IfcLabel/IfcText ที่มี wrappedValue
                                        val = getattr(nv, "wrappedValue", None)
                                        if val is None:
                                            val = safe(nv)
                                    pset_items.append(f"{pset_name}.{pname}={safe(val)}")
            except Exception:
                pass

            if pset_items:
                line.append(" | ".join(pset_items[:20]))  # จำกัด 20 รายการต่อ element
            parts.append(" ; ".join(line))

    return clean_text("\n".join(parts))
# ---------- /IFC ----------

def create_vector_store():
    # ---------- IFC: รวบรวมไฟล์ .ifc ในโฟลเดอร์ Data ----------
    ifc_files = glob("Data/**/*.ifc", recursive=True)
    if not ifc_files:
        print("ไม่พบไฟล์ IFC ในโฟลเดอร์ Data (เช่น Data/xxx.ifc)")
        return
    file_paths = [(p, os.path.splitext(os.path.basename(p))[0]) for p in ifc_files]
    # -----------------------------------------------------------

    docs = []
    for path, source_tag in file_paths:
        # ---------- IFC: แปลง IFC เป็นข้อความ ----------
        cleaned = extract_ifc_text(path)
        doc = Document(
            page_content=cleaned,
            metadata={
                "page": 1,  # คงคีย์เดิมไว้ (ไม่มีหน้าเหมือน PDF จึงกำหนดเป็น 1)
                "source": source_tag,
                "file": os.path.basename(path),
            }
        )
        docs.append(doc)

    print(f"โหลดเอกสารทั้งหมด {len(docs)} หน้า")

    # แบ่งเอกสารเป็น Chunks (คงค่าตามเดิม)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""],
        is_separator_regex=False,
    )
    splits = text_splitter.split_documents(docs)
    print(f"แบ่งเอกสารออกเป็น {len(splits)} chunks")

    # สร้าง Embeddings (คงเดิม)
    embedding = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    # สร้าง Vector Store จากเอกสาร (คงเดิม)
    vectordb = FAISS.from_documents(
        documents=splits,
        embedding=embedding
    )

    # บันทึก Vector Store (คงเดิม)
    vectordb.save_local(VECTOR_STORE_PATH)
    print("เรียบร้อยแล้ว")

if __name__ == "__main__":
    create_vector_store()
