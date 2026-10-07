import json
import os


def load_coref_data(file_path):
    """Load coreference documents from a JSON array, object, or JSONL file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"ไม่พบไฟล์ชุดข้อมูล: {file_path}")

    with open(file_path, "r", encoding="utf-8") as data_file:
        content = data_file.read()

    if not content.strip():
        return []

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        data = []
        for line_number, line in enumerate(content.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"JSON ผิดรูปแบบใน {file_path} บรรทัดที่ "
                    f"{line_number}: {exc.msg}"
                ) from exc
            if not isinstance(record, dict):
                raise ValueError(
                    f"ข้อมูลใน {file_path} บรรทัดที่ {line_number} "
                    "ต้องเป็น JSON object"
                )
            data.append(record)
        return data

    if isinstance(parsed, dict):
        return [parsed]
    if isinstance(parsed, list) and all(isinstance(item, dict) for item in parsed):
        return parsed
    raise ValueError(
        f"ข้อมูลใน {file_path} ต้องเป็น JSON object, "
        "JSON array ของ object หรือ JSONL"
    )


def _validate_record(record, record_number):
    text = record.get("text")
    clusters = record.get("clusters")
    if not isinstance(text, str) or not text:
        raise ValueError(f"ตัวอย่างที่ {record_number}: ต้องมี text ที่ไม่ว่าง")
    if not isinstance(clusters, list):
        raise ValueError(f"ตัวอย่างที่ {record_number}: clusters ต้องเป็น list")

    cluster_strings = record.get("clusters_strings")
    if cluster_strings is not None and (
        not isinstance(cluster_strings, list)
        or len(cluster_strings) != len(clusters)
    ):
        raise ValueError(
            f"ตัวอย่างที่ {record_number}: clusters_strings "
            "ต้องมีจำนวน cluster เท่ากับ clusters"
        )

    seen_spans = set()
    for cluster_index, cluster in enumerate(clusters):
        if not isinstance(cluster, list) or not cluster:
            raise ValueError(
                f"ตัวอย่างที่ {record_number}: cluster {cluster_index} "
                "ต้องมี mention อย่างน้อยหนึ่งช่วง"
            )
        if cluster_strings is not None and (
            not isinstance(cluster_strings[cluster_index], list)
            or len(cluster_strings[cluster_index]) != len(cluster)
        ):
            raise ValueError(
                f"ตัวอย่างที่ {record_number}: clusters_strings "
                f"ของ cluster {cluster_index} มีจำนวนไม่ตรงกัน"
            )

        for span_index, span in enumerate(cluster):
            if (
                not isinstance(span, list)
                or len(span) != 2
                or type(span[0]) is not int
                or type(span[1]) is not int
            ):
                raise ValueError(
                    f"ตัวอย่างที่ {record_number}: span {cluster_index}:"
                    f"{span_index} ต้องเป็น [start, end] จำนวนเต็ม"
                )
            start, end = span
            if not 0 <= start < end <= len(text):
                raise ValueError(
                    f"ตัวอย่างที่ {record_number}: span {span} "
                    f"อยู่นอกขอบเขตข้อความ (ยาว {len(text)} อักขระ)"
                )
            if (start, end) in seen_spans:
                raise ValueError(
                    f"ตัวอย่างที่ {record_number}: พบ span ซ้ำ {(start, end)}"
                )
            seen_spans.add((start, end))
            if (
                cluster_strings is not None
                and text[start:end] != cluster_strings[cluster_index][span_index]
            ):
                raise ValueError(
                    f"ตัวอย่างที่ {record_number}: text[{start}:{end}] "
                    "ไม่ตรงกับ clusters_strings"
                )


def prepare_coref_examples(records, tokenizer, max_length, overlap):
    """Tokenize documents into overlapping windows while retaining gold clusters."""
    special_token_count = tokenizer.num_special_tokens_to_add(pair=False)
    content_length = max_length - special_token_count
    if content_length <= 0:
        raise ValueError("max_length ต้องมากกว่าจำนวน special tokens ของ tokenizer")
    if not 0 <= overlap < content_length:
        raise ValueError("overlap ต้องไม่น้อยกว่า 0 และน้อยกว่า content length")

    examples = []
    for record_number, record in enumerate(records, start=1):
        _validate_record(record, record_number)
        encoded = tokenizer(
            record["text"],
            add_special_tokens=True,
            return_offsets_mapping=True,
            return_special_tokens_mask=True,
            return_attention_mask=True,
            return_overflowing_tokens=True,
            truncation=True,
            max_length=max_length,
            stride=overlap,
        )
        window_ids = encoded["input_ids"]
        window_offsets = encoded["offset_mapping"]
        window_special_masks = encoded["special_tokens_mask"]
        window_attention_masks = encoded["attention_mask"]
        if not window_ids:
            raise ValueError(f"ตัวอย่างที่ {record_number}: ไม่พบ token ในข้อความ")

        token_offsets = set()
        for input_ids, offsets, special_mask, attention_mask in zip(
            window_ids,
            window_offsets,
            window_special_masks,
            window_attention_masks,
        ):
            if not (
                len(input_ids)
                == len(offsets)
                == len(special_mask)
                == len(attention_mask)
            ):
                raise ValueError(
                    f"ตัวอย่างที่ {record_number}: tokenizer คืนข้อมูล token ไม่ครบ"
                )
            token_offsets.update(
                (start, end)
                for (start, end), is_special in zip(offsets, special_mask)
                if not is_special and end > start
            )
        if not token_offsets:
            raise ValueError(f"ตัวอย่างที่ {record_number}: ไม่พบ token ในข้อความ")

        token_spans = []
        for cluster_index, cluster in enumerate(record["clusters"]):
            for start, end in cluster:
                covered = [
                    (token_start, token_end)
                    for token_start, token_end in token_offsets
                    if token_end > start and token_start < end
                ]
                if not covered:
                    raise ValueError(
                        f"ตัวอย่างที่ {record_number}: จัดแนว span "
                        f"[{start}, {end}] กับ token ไม่ได้"
                    )
                if min(token_start for token_start, _ in covered) > start or max(
                    token_end for _, token_end in covered
                ) < end:
                    raise ValueError(
                        f"ตัวอย่างที่ {record_number}: tokenizer ครอบคลุม "
                        f"span [{start}, {end}] ไม่ครบ"
                    )
                token_spans.append((start, end, cluster_index))

        covered_spans = set()
        for input_ids, attention_mask, offsets, special_mask in zip(
            window_ids,
            window_attention_masks,
            window_offsets,
            window_special_masks,
        ):
            mentions = []
            for span_index, (start, end, cluster_index) in enumerate(token_spans):
                covered = [
                    position
                    for position, ((token_start, token_end), is_special) in enumerate(
                        zip(offsets, special_mask)
                    )
                    if not is_special
                    and token_end > start
                    and token_start < end
                    and token_end > token_start
                ]
                if not covered:
                    continue
                first, last = covered[0], covered[-1]
                if offsets[first][0] > start or offsets[last][1] < end:
                    continue
                mentions.append((first, last, cluster_index))
                covered_spans.add(span_index)

            examples.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "mentions": mentions,
                }
            )

        if len(covered_spans) != len(token_spans):
            missing = len(token_spans) - len(covered_spans)
            raise ValueError(
                f"ตัวอย่างที่ {record_number}: มี {missing} span ยาวเกิน "
                "หน้าต่าง token; ลด overlap หรือเพิ่ม max_length"
            )

    return examples
