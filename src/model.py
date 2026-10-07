import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

class ZCorefModel(nn.Module):
    def __init__(self, model_name="airesearch/wangchanberta-base-att-spm-uncased"):
        super(ZCorefModel, self).__init__()
        # โหลด Config และ Pre-trained Model ภาษาไทย
        self.config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name, config=self.config)
        
        # Hidden size ของโมเดล (เช่น 768 สำหรับตระกูล BERT/WangchanBERTa)
        hidden_size = self.config.hidden_size
        
        # Heads for mention boundaries and antecedent ranking.
        self.span_start_classifier = nn.Linear(hidden_size, 1)
        self.span_end_classifier = nn.Linear(hidden_size, 1)
        self.mention_projection = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.distance_embedding = nn.Embedding(10, 20)
        self.antecedent_scorer = nn.Sequential(
            nn.Linear(hidden_size * 3 + 20, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 1),
        )
        self.dummy_antecedent_score = nn.Parameter(torch.zeros(()))
        
    def forward(self, input_ids, attention_mask, mention_spans=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state

        start_scores = self.span_start_classifier(sequence_output)
        end_scores = self.span_end_classifier(sequence_output)

        if mention_spans is None:
            return start_scores, end_scores
        if input_ids.shape[0] != 1:
            raise ValueError("การคำนวณ antecedent รองรับเอกสารเดียวต่อครั้ง")

        antecedent_scores = self.score_antecedents(
            sequence_output[0], mention_spans
        )
        return start_scores, end_scores, antecedent_scores

    def score_antecedents(self, sequence_output, mention_spans):
        mention_count = mention_spans.shape[0]
        if mention_count == 0:
            return sequence_output.new_empty((0, 1))

        start_vectors = sequence_output[mention_spans[:, 0]]
        end_vectors = sequence_output[mention_spans[:, 1]]
        mentions = self.mention_projection(
            torch.cat((start_vectors, end_vectors), dim=-1)
        )

        current = mentions.unsqueeze(1).expand(-1, mention_count, -1)
        antecedent = mentions.unsqueeze(0).expand(mention_count, -1, -1)
        positions = mention_spans[:, 0]
        distance = (positions.unsqueeze(1) - positions.unsqueeze(0)).abs()
        distance_buckets = torch.floor(torch.log2(distance.clamp(min=1).float()))
        distance_buckets = distance_buckets.long().clamp(max=9)
        distance_vectors = self.distance_embedding(distance_buckets)

        pair_features = torch.cat(
            (
                current,
                antecedent,
                current * antecedent,
                distance_vectors,
            ),
            dim=-1,
        )
        pair_scores = self.antecedent_scorer(pair_features).squeeze(-1)
        invalid_antecedents = torch.triu(
            torch.ones(
                mention_count,
                mention_count,
                dtype=torch.bool,
                device=sequence_output.device,
            )
        )
        pair_scores = pair_scores.masked_fill(invalid_antecedents, -1e4)
        dummy_scores = self.dummy_antecedent_score.expand(mention_count, 1)
        return torch.cat((dummy_scores, pair_scores), dim=1)

if __name__ == "__main__":
    # ทดลองสร้าง Object ของโมเดล
    print("กำลังโหลดโมเดล Encoder ภาษาไทย...")
    model = ZCorefModel()
    print("โหลดโครงสร้าง ZCorefModel สำเร็จพร้อมใช้งาน!")