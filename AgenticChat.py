import os
from dotenv import load_dotenv
load_dotenv()
Gemini_key = os.getenv("Gemini_key")

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain.retrievers.multi_query import MultiQueryRetriever
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from langchain_google_genai import ChatGoogleGenerativeAI

VECTOR_STORE_PATH = "Data/vector_store"

def setup_components():
    #model_name = "openthaigpt/openthaigpt1.5-7b-instruct"
    #tokenizer = AutoTokenizer.from_pretrained(model_name)
    #model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
    #pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    #llm = HuggingFacePipeline(pipeline=pipe)

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0,
        google_api_key=Gemini_key
    )

    if not os.path.exists(VECTOR_STORE_PATH):
        print(f"ไม่พบ Vector Store ที่ '{VECTOR_STORE_PATH}'")
        return None, None, None

    embedding = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    vectordb = FAISS.load_local(VECTOR_STORE_PATH, embedding, allow_dangerous_deserialization=True)

    multi_query_retriever = MultiQueryRetriever.from_llm(
        retriever=vectordb.as_retriever(search_kwargs={"k": 12}),
        llm=llm
    )
    return llm, vectordb, multi_query_retriever

def ask_llm(llm, prompt):
    try:
        response = llm.invoke(prompt)
        return response.content.strip()
    except AttributeError:
        return str(response).strip()

# ---------- IFC: เปลี่ยนสรุปให้เป็นภาพรวมโมเดลอาคาร ----------
def summarize_subject_structure(llm, texts):
    joined = "\n".join(texts)
    prompt = f"""ข้อมูลจากไฟล์ IFC (แปลงเป็นข้อความแล้ว):
----
{joined}
----
สรุปภาพรวมโมเดล:
1) ชื่อโครงการ/อาคาร (ถ้ามี)
2) จำนวนชั้น (Building Storeys) และรายชื่อ
3) จำนวนองค์ประกอบหลัก (เช่น กำแพง ประตู หน้าต่าง เสา คาน พื้น ห้อง/สเปซ)
4) ข้อมูลเด่น (เช่น พื้นที่หรือปริมาตรถ้ามีใน PropertySet)
ตอบให้กระชับ ชัดเจน เป็นหัวข้อย่อยภาษาไทย"""
    return ask_llm(llm, prompt)
# ---------- /IFC ----------

def verify_answer(llm, question, answer, context):
    verify_prompt = f"""
    ตรวจสอบคำตอบนี้ถูกต้องหรือไม่?
    คำถาม: {question}
    คำตอบ: {answer}
    เอกสารและประวัติที่เกี่ยวข้อง: {context}
    - ถ้ามีข้อมูลสนับสนุนให้ตอบ "YES"
    - ถ้าไม่มีข้อมูลสนับสนุน หรือผิด ให้ตอบ "NO"
    """
    result = ask_llm(llm, verify_prompt).upper()
    return result.startswith("YES")

def main():
    llm, vectordb, multi_query_retriever = setup_components()
    if not llm:
        return

    message_history = []

    print("\nพร้อมแล้ว (พิมพ์ quit เพื่อออก)")
    while True:
        try:
            question = input("you: ")
            if question.lower() == "quit":
                print("BOT: พบกันใหม่นะ!")
                break

            previous_conversation = "\n".join([
                f"ผู้ใช้: {msg[1]}" if msg[0] == "user" else f"BOT: {msg[1]}"
                for msg in message_history[-10:]
            ])
            planning_prompt = f"""
                บทสนทนาก่อนหน้า:
                {previous_conversation}

                วางแผนการทำงานเป็นลำดับ:
                1. จะค้นหาอะไรเป็นอันดับแรก? (โครงสร้างอาคาร, ชั้น, ชนิดองค์ประกอบ, PropertySet)
                2. ถ้าไม่เจอข้อมูล จะทำอย่างไร? (ขยายคำค้น/ถามใหม่)
                3. ถ้าเจอข้อมูล จะสรุปอย่างไร? (อ้างอิงข้อมูลจากบริบทด้านล่าง)
                คำถาม: {question}
                """
            plan = ask_llm(llm, planning_prompt)

            retrieved_docs = multi_query_retriever.invoke(question)
            retrieved_texts = [doc.page_content for doc in retrieved_docs]
            combined_context = previous_conversation + "\n" + "\n".join(retrieved_texts)

            # ---------- IFC: เงื่อนไขเรียกสรุปภาพรวมโมเดล ----------
            lower_q = question.lower()
            if any(kw in lower_q for kw in [
                "สรุป", "ภาพรวม", "storey", "ชั้น", "จำนวน", "องค์ประกอบ", "กำแพง",
                "ประตู", "หน้าต่าง", "เสา", "คาน", "พื้น", "space", "พื้นที่", "ปริมาตร", "ifc"
            ]):
                answer = summarize_subject_structure(llm, retrieved_texts)
            else:
                direct_prompt = f"""
                บทสนทนาก่อนหน้า:
                {previous_conversation}

                แผนการทำงานล่าสุด:
                {plan}
                คำถาม: {question}
                จากข้อมูลทั้งหมด (สกัดจากไฟล์ IFC):
                {combined_context}

                ช่วยตอบให้ละเอียดได้ใจความครบถ้วน โดยอ้างอิงเฉพาะสิ่งที่พบในบริบทด้านบน
                ถ้าไม่พบข้อมูล ให้ตอบว่า 'ไม่พบข้อมูล'
                """
                answer = ask_llm(llm, direct_prompt)
            # ---------- /IFC ----------

            message_history.append(("user", question))
            message_history.append(("ai", answer))

            attempts = 0
            while attempts < 2:
                if verify_answer(llm, question, answer, combined_context):
                    print(f"BOT: {answer}")
                    break
                else:
                    attempts += 1
                    retrieved_docs = multi_query_retriever.invoke(question)
                    retrieved_texts = [doc.page_content for doc in retrieved_docs]
                    combined_context = previous_conversation + "\n" + "\n".join(retrieved_texts)
                    answer = ask_llm(llm, f"""
                        บทสนทนาก่อนหน้า:
                        {previous_conversation}
                        แผนการทำงานล่าสุด:
                        {plan}
                        คำถาม: {question}
                        จากข้อมูลใหม่ (สกัดจากไฟล์ IFC):
                        {combined_context}
                        ตอบอีกครั้ง ถ้าไม่พบข้อมูล ให้บอกว่า 'ไม่พบข้อมูล'
                        """)
                    message_history.append(("ai", answer))
            else:
                print("BOT: คำตอบอาจไม่ตรงตามเอกสารหรือประวัติทั้งหมด")

        except Exception as e:
            print(f"เกิดข้อผิดพลาด: {e}")

if __name__ == "__main__":
    main()
