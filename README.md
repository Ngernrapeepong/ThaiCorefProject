# ThaiCoref

ชุดข้อมูล `data/train.json` ใช้รูปแบบ JSONL: หนึ่งเอกสารต่อบรรทัด มี `text`
และ `clusters` ซึ่งแต่ละช่วงเขียนเป็น `[start, end]` แบบ character offset
(รวม start ไม่รวม end) โดย index ใช้ Python Unicode code point
`clusters_strings` เป็นข้อมูลตรวจสอบเสริมได้

สคริปต์ฝึกจะใช้ WangchanBERTa เป็น encoder และเรียนรู้ทั้งตำแหน่งเริ่ม/จบของ
mentions กับการเลือก antecedent เพื่อจัดกลุ่ม coreference เอกสารที่ยาวกว่า
ความยาวสูงสุดจะถูกแบ่งเป็นหน้าต่างที่ซ้อนทับกัน; การเชื่อมโยงระหว่าง mentions
ที่ไม่ปรากฏร่วมกันในหน้าต่างเดียวจะไม่ถูกเรียนรู้ในตัวอย่างนั้น

```powershell
.\.venv\Scripts\python.exe train.py
```

กำหนดชุด validation ได้ด้วย `--dev-data path\to\dev.json`; หากไม่ระบุหรือไฟล์
ว่าง จะใช้ค่า training loss เพื่อเลือก checkpoint แต่ละ epoch
ไฟล์ checkpoint และ tokenizer จะถูกบันทึกใน `outputs/thai-coref` โดยค่าเริ่มต้น

ตัวเลือกหลัก: `--epochs`, `--learning-rate`, `--max-length`, `--overlap`,
`--model-name` และ `--output-dir` ใช้ `train.py --help` เพื่อดูรายการทั้งหมด
