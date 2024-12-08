import sys
import os
import string
import time
import re
from tqdm import tqdm
from sentence_splitter import split_text_into_sentences
from nltk.tokenize import sent_tokenize
import nltk
import transformers
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification
import pandas as pd
from transformers import BertTokenizer, BertModel
import torch
from pathlib import Path
from torchtext.vocab import GloVe, FastText
from torchtext.data import get_tokenizer
from sentence_transformers import SentenceTransformer, util
from openai import AzureOpenAI

# current_dir = os.path.dirname(os.path.abspath(__file__))
# grandparent_dir = os.path.join(os.path.dirname(current_dir), "erevise/models")
# if current_dir not in sys.path:
#     sys.path.append(grandparent_dir)
#
# print("Python version:", sys.version)
# print("Executable:", sys.executable)
# print("sys.path:", sys.path)

from checkpoints.bertalign.bertalign import Bertalign

nltk.download('punkt')

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

BASE_DIR = Path(__file__).resolve().parent.parent

client = AzureOpenAI(
        api_key="",
        api_version="2023-12-01-preview",
        azure_endpoint="https://erevise-us2.openai.azure.com/"
    )



class TextTwinModel(torch.nn.Module):
    def __init__(self, out_dim=1, freeze_bert=False):
        super(TextTwinModel, self).__init__()
        self.out_dim = out_dim
        self.bert_model = transformers.BertModel.from_pretrained("bert-base-cased")
        if freeze_bert:
            for param in self.bert_model.parameters():
                param.requires_grad = False

        self.fc = torch.nn.Sequential(
            torch.nn.BatchNorm1d(768*2, eps=1e-05, momentum=0.1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.5, inplace=False),
            torch.nn.Linear(768*2, out_dim))

    def forward(self, input_ids1, attention_mask1, input_ids2, attention_mask2):

        _, out1 = self.bert_model(input_ids= input_ids1, attention_mask=attention_mask1, return_dict=False)
        _, out2 = self.bert_model(input_ids= input_ids2, attention_mask=attention_mask2, return_dict=False)
        out = torch.concat((out1, out2), dim=1)
        # out = F.normalize(out, dim=-1)
        out = self.fc(out)
        return out

class DesirableModel(torch.nn.Module):
    def __init__(self, out_dim=1, freeze_bert=False):
        super(DesirableModel, self).__init__()
        embedding_dim = 768

        self.model = AutoModelForSequenceClassification.from_pretrained("distilroberta-base")

        for param in self.model.parameters():
            if freeze_bert:
                param.requires_grad = False
            else:
                param.requires_grad = True

        self.fc = torch.nn.Sequential(
            torch.nn.BatchNorm1d(embedding_dim, eps=1e-05, momentum=0.1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.5, inplace=False),
            torch.nn.Linear(embedding_dim, out_dim))


    def forward(self, input_ids, attention_mask):
        hidden_out = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True).hidden_states[-1]
        out = torch.mean(hidden_out, dim=1)
        out = F.normalize(out, dim=-1)
        out = self.fc(out)
        return out
class ScoreRevision():
    def __init__(self, draft1, draft2, topic="MVP"):
        self.draft1 = draft1
        self.draft2 = draft2
        self.topic = topic
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.surface_content_classifer_model = TextTwinModel()
        self.successfulness_classifer_model = DesirableModel()

        checkpoint = torch.load('./checkpoints/model_coarse_classifier.pkl', map_location=torch.device(self.device))
        self.surface_content_classifer_model.load_state_dict(checkpoint, strict=False)
        self.surface_content_classifer_model.to(self.device)
        self.surface_content_classifer_model.eval()

        checkpoint = torch.load('./checkpoints/model_successfulness.pth', map_location=torch.device(self.device))
        self.successfulness_classifer_model.load_state_dict(checkpoint, strict=False)
        self.successfulness_classifer_model.to(self.device)
        self.successfulness_classifer_model.eval()

        self.sentence_encoder_model = SentenceTransformer("LaBSE")
        self.tokenizer_align = BertTokenizer.from_pretrained('bert-base-cased')
        self.tokenizer_success = AutoTokenizer.from_pretrained("sentence-transformers/all-distilroberta-v1")

        self.master_df = None

    ############### sentence alignment
    def align_document(self):

        old_draft_index_list = []
        old_draft_aligned_index_list = []
        old_draft_sentence_list = []

        new_draft_index_list = []
        new_draft_aligned_index_list = []
        new_draft_sentence_list = []

        old_dft = self.draft1
        new_dft = self.draft2
        old_dft = self.clean_text(old_dft)
        new_dft = self.clean_text(new_dft)

        old_draft_list_tmp = split_text_into_sentences(text=old_dft, language="en")
        new_draft_list_tmp = split_text_into_sentences(text=new_dft, language="en")
        old_draft_list_tmp = [sent_tokenize(string) for string in old_draft_list_tmp]
        new_draft_list_tmp = [sent_tokenize(string) for string in new_draft_list_tmp]
        old_draft_list = []
        new_draft_list = []
        for tmp in old_draft_list_tmp:
            old_draft_list += tmp
        for tmp in new_draft_list_tmp:
            new_draft_list += tmp

        # sentence_splitter is better than nltk sent_tokenizer
        # old_draft_list = sent_tokenize(old_dft)
        # new_draft_list = sent_tokenize(new_dft)

        src_align_result, tgt_align_result = self.get_sentence_alignment(old_draft_list, new_draft_list)
        print("\nOld sentence ids: ", src_align_result)
        print("New sentence ids: ", tgt_align_result)

        old_sent_length = len(src_align_result)
        new_sent_length = len(tgt_align_result)

        i = j = i_count = j_count = 0

        while i < old_sent_length or j < new_sent_length:
            # convert to new index
            if i < old_sent_length and j < new_sent_length and src_align_result[i] == j and tgt_align_result[
                j] == i:
                old_draft_index_list.append(i_count)
                old_draft_aligned_index_list.append(src_align_result[i])
                old_draft_sentence_list.append(old_draft_list[i])

                new_draft_index_list.append(j_count)
                new_draft_aligned_index_list.append(tgt_align_result[j])
                new_draft_sentence_list.append(new_draft_list[j])

                i += 1
                j += 1
                i_count += 1
                j_count += 1

            elif i < old_sent_length and j < new_sent_length and src_align_result[i] == "DELETE" and \
                    tgt_align_result[
                        j] == "ADD":
                old_draft_index_list.append(i_count)
                old_draft_aligned_index_list.append(src_align_result[i])
                old_draft_sentence_list.append(old_draft_list[i])

                new_draft_index_list.append("")
                new_draft_aligned_index_list.append("")
                new_draft_sentence_list.append("")
                i += 1
                i_count += 1

                old_draft_index_list.append("")
                old_draft_aligned_index_list.append("")
                old_draft_sentence_list.append("")

                new_draft_index_list.append(j_count)
                new_draft_aligned_index_list.append(tgt_align_result[j])
                new_draft_sentence_list.append(new_draft_list[j])
                j += 1
                j_count += 1


            elif i < old_sent_length and src_align_result[i] == "DELETE":
                old_draft_index_list.append(i_count)
                old_draft_aligned_index_list.append(src_align_result[i])
                old_draft_sentence_list.append(old_draft_list[i])

                new_draft_index_list.append("")
                new_draft_aligned_index_list.append("")
                new_draft_sentence_list.append("")
                i += 1
                i_count += 1

            elif j < new_sent_length and tgt_align_result[j] == "ADD":
                old_draft_index_list.append("")
                old_draft_aligned_index_list.append("")
                old_draft_sentence_list.append("")

                new_draft_index_list.append(j_count)
                new_draft_aligned_index_list.append(tgt_align_result[j])
                new_draft_sentence_list.append(new_draft_list[j])
                j += 1
                j_count += 1
            else:
                raise "error"

        df = pd.DataFrame({"old_sentence_id": old_draft_index_list,
                           "old_sentence_aligned_id": old_draft_aligned_index_list,
                           "old_sentence": old_draft_sentence_list,
                           "new_sentence": new_draft_sentence_list,
                           "new_sentence_id": new_draft_index_list,
                           "new_sentence_aligned_id": new_draft_aligned_index_list,
                           })

        df = self.add_sentence_pair_labels(df)
        # if not os.path.exists(f"./outputs/{name}"):
        #     os.makedirs(f"./outputs/{name}")
        # df.to_excel(f"./outputs/{name}/{e_id}.xlsx", index=False)
        print(df)

        self.master_df = df

        return df

    def get_text_similarity(self, text1, text2):
        embed1 = self.sentence_encoder_model.encode(text1)
        embed2 = self.sentence_encoder_model.encode(text2)

        simi = util.dot_score(embed1, embed2)
        simi = simi.squeeze().tolist()
        return simi

    def clean_text(self, text):
        text = re.sub(r'[\n|\r]', ' ', text)
        # remove space between ending word and punctuations
        text = re.sub(r'[ ]+([\.\?\!\,]{1,})', r'\1 ', text)
        # remove duplicated spaces
        text = re.sub(r' +', ' ', text)
        # add space if no between punctuation and words
        text = re.sub(r'([a-z|A-Z]{2,})([\.\?\!]{1,})([a-z|A-Z]{1,})', r'\1\2\n\3', text)
        # handle case "...\" that" that the sentence spliter cannot do
        text = re.sub(r'([\?\!\.]+)(\")([\s]+)([a-z|A-Z]{1,})', r'\1\2\n\3\4', text)
        # remove space between letter and punctuation
        text = re.sub(r'([a-z|A-Z]{2,})([ ]+)([\.\?\!])', r'\1\3', text)
        # handle case '\".word' that needs space after '.'
        text = re.sub(r'([\"\']+\.)([a-z|A-Z]{1,})', r'\1\n\2', text)
        # handle case '.\"word' that needs space after '\"'
        text = re.sub(r'(\.[\"\']+)([a-z|A-Z]{1,})', r'\1\n\2', text)
        # text = re.sub('\n', ' ', text)
        text = text.strip()
        text = text.lower()
        return text

    def preprocess_essay(self, text):
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = text.replace("\n", " ")
        text = text.replace('"', '')
        text = text.replace("'", "")
        text = text.replace("“", "")
        text = text.replace("x000d", "")
        text = text.replace("x000D", "")
        text = " ".join([token for token in text.split(" ")])
        text = text.strip()
        text = text.lower()
        text = text.split()
        return text

    def get_sentence_alignment(self, old_draft_list, new_draft_list):
        old_draft_list = [re.sub(r'[^\w\s]', '', s) + "." for s in old_draft_list]
        new_draft_list = [re.sub(r'[^\w\s]', '', s) + "." for s in new_draft_list]

        # if len(new_draft_list) == 1:
        #     new_draft_list.append("empty.")

        old_doc = " \n\n".join(old_draft_list)
        new_doc = " \n\n".join(new_draft_list)

        src = old_doc.replace("\"", "").replace("\'", "").replace("...", " ")
        tgt = new_doc.replace("\"", "").replace("\'", "").replace("...", " ")

        aligner = Bertalign(src, tgt, skip=-0.05, is_split=True)
        aligner.align_sents()
        src_sents = aligner.src_sents
        tgt_sents = aligner.tgt_sents
        result = aligner.result

        src_align_result = []
        tgt_align_result = []
        for i in range(len(result)):
            src_sent_idx_list = result[i][0]
            tgt_sent_idx_list = result[i][1]
            print("\nAligned sentences:", src_sent_idx_list, tgt_sent_idx_list)
            print("Old sentences:    ", " ".join([src_sents[k] for k in src_sent_idx_list]))
            print("New sentences:    ", " ".join([tgt_sents[k] for k in tgt_sent_idx_list]))

            # if return one-on-one aligment
            if len(src_sent_idx_list) == 1 and len(tgt_sent_idx_list) == 1:
                tmp_sent_src = src_sents[src_sent_idx_list[0]]
                tmp_sent_tgt = tgt_sents[tgt_sent_idx_list[0]]
                tmp_similarity = self.get_text_similarity(tmp_sent_src, tmp_sent_tgt)
                # in case the matched two are not aligned in fact
                print("Similarity score: ", tmp_similarity)
                if tmp_similarity <= 0.5:
                    src_align_result.append("DELETE")
                    tgt_align_result.append("ADD")
                else:
                    src_align_result.append(tgt_sent_idx_list[0])
                    tgt_align_result.append(src_sent_idx_list[0])

            # if return one-on-zero alignment
            elif len(src_sent_idx_list) < len(tgt_sent_idx_list) and len(src_sent_idx_list) == 0:
                for j in range(len(tgt_sent_idx_list)):
                    tgt_align_result.append("ADD")

            # if return zero-on-one alignment
            elif len(src_sent_idx_list) > len(tgt_sent_idx_list) and len(tgt_sent_idx_list) == 0:
                for j in range(len(src_sent_idx_list)):
                    src_align_result.append("DELETE")

            # multiple-on-multiple alignment
            else:
                src_mask = [-1] * len(src_sent_idx_list)
                tgt_mask = [-1] * len(tgt_sent_idx_list)

                for j in range(len(src_sent_idx_list)):

                    tmp_idx = 0
                    tmp_sent_src = src_sents[src_sent_idx_list[j]]
                    max_similarity = float("-inf")

                    for k in range(len(tgt_sent_idx_list)):
                        if tgt_mask[k] != -1:
                            continue
                        tmp_sent_tgt = tgt_sents[tgt_sent_idx_list[k]]
                        tmp_similarity = self.get_text_similarity(tmp_sent_src, tmp_sent_tgt)
                        if tmp_similarity > max_similarity:
                            max_similarity = tmp_similarity
                            tmp_idx = k
                    if max_similarity > 0.85:
                        src_align_result.append(tgt_sent_idx_list[tmp_idx])
                        src_mask[j] = tgt_sent_idx_list[tmp_idx]
                        tgt_mask[tmp_idx] = src_sent_idx_list[j]
                    else:
                        src_align_result.append("DELETE")

                for j in range(len(tgt_sent_idx_list)):
                    if tgt_mask[j] != -1:
                        tgt_align_result.append(tgt_mask[j])
                        continue

                    tmp_idx = 0
                    tmp_sent_tgt = tgt_sents[tgt_sent_idx_list[j]]
                    max_similarity = float("-inf")

                    for k in range(len(src_sent_idx_list)):
                        if src_mask[k] != -1:
                            continue
                        tmp_sent_src = src_sents[src_sent_idx_list[k]]
                        tmp_similarity = self.get_text_similarity(tmp_sent_tgt, tmp_sent_src)
                        if tmp_similarity > max_similarity:
                            max_similarity = tmp_similarity
                            tmp_idx = k
                    if max_similarity > 0.85:
                        tgt_align_result.append(src_sent_idx_list[tmp_idx])
                    else:
                        tgt_align_result.append("ADD")

            # print(src_align_result)
            # print(tgt_align_result)

        return src_align_result, tgt_align_result

    def add_sentence_pair_labels(self, df):
        label_list = []
        old_sent_list = df["old_sentence"]
        new_sent_list = df["new_sentence"]
        for i in range(len(old_sent_list)):
            old_sent = old_sent_list[i]
            new_sent = new_sent_list[i]
            if old_sent == new_sent:
                label_list.append("")
            else:
                label = self.get_model_prediction(old_sent, new_sent)
                label_list.append(label)
        df["coarse_label"] = label_list
        return df

    def get_model_prediction(self, old_sent, new_sent):
        if len(old_sent) == 0:
            old_sent = "empty."
        if len(new_sent) == 0:
            new_sent = "empty."

        old_sent = re.sub(r'[^\w\s]', '', old_sent) + "."
        new_sent = re.sub(r'[^\w\s]', '', new_sent) + "."

        old_input = self.tokenizer_align(old_sent, padding='max_length', max_length=512, truncation=True, return_tensors="pt")
        new_input = self.tokenizer_align(new_sent, padding='max_length', max_length=512, truncation=True, return_tensors="pt")
        old_mask = old_input['attention_mask'].to(self.device)
        old_input_id = old_input['input_ids'].squeeze(1).to(self.device)

        new_mask = new_input['attention_mask'].to(self.device)
        new_input_id = new_input['input_ids'].squeeze(1).to(self.device)

        with torch.no_grad():
            logits = self.surface_content_classifer_model(old_input_id, old_mask, new_input_id, new_mask)
            outputs = torch.sigmoid(logits)
            predict = outputs.squeeze().cpu().round().tolist()
            if predict == 1:
                predict = "content"
            else:
                predict = "surface"
            return predict

    ################ argument context

    def get_argument_context(self):

        print("get argument context...")

        first_draft = self.draft1
        second_draft = self.draft2

        first_draft = self.clean_text(first_draft)
        second_draft = self.clean_text(second_draft)

        old_argument_context = self.get_argument_context_chatgpt(first_draft)
        new_argument_context = self.get_argument_context_chatgpt(second_draft)

        self.master_df["old_argument_context"] = old_argument_context
        self.master_df["new_argument_context"] = new_argument_context


    def get_argument_context_chatgpt(self, text):
        try:
            context_extract = f"list evidence and reasoning sentences in bullet points. \n\n{text}"


            agent = client.chat.completions.create(
                model="gpt-35-turbo",  # model = "gpt-4, gpt-35-turbo"
                temperature=0,
                messages=[
                    # {"role": "system", "content": context_extract},
                          {"role": "user", "content": context_extract}])

            extract = agent.choices[0].message.content

            context_summarize = f"summarize in two sentences. \n\n{extract}"
            agent = client.chat.completions.create(
                model="gpt-35-turbo",  # model = "gpt-4, gpt-35-turbo"
                temperature=0,
                messages=[
                    # {"role": "system", "content": context_summarize},
                          {"role": "user", "content": context_summarize}])
            argument_context = agent.choices[0].message.content
        except:
            argument_context = ""
        return argument_context

    ################ predict successful revisions
    def predict_successfulness(self):
        print("predict successful revisions...")
        df_master = self.master_df
        old_sentences = df_master["old_sentence"].tolist()
        new_sentences = df_master["new_sentence"].tolist()

        old_contexts = df_master["old_argument_context"].tolist()
        new_contexts = df_master["new_argument_context"].tolist()


        revision_list = []
        context_list = []
        for i in range(len(old_sentences)):
            old_sent = old_sentences[i]
            new_sent = new_sentences[i]
            old_ac = old_contexts[i]
            new_ac = new_contexts[i]
            if str(old_sent) == '':
                action = "ADD"
                context_list.append(new_ac)
                revision_list.append(new_sent)
            elif str(new_sent) == '':
                action = "DELETE"
                context_list.append(old_ac)
                revision_list.append(old_sent)
            else:
                action = "MODIFY"
                context_list.append(new_ac)
                revision_list.append(new_sent)

        df_master["used_context"] = context_list
        df_master["used_revision"] = revision_list

        successful_list = []
        coarse_labels = df_master["coarse_label"].tolist()
        for i in range(len(coarse_labels)):
            coarse_label = coarse_labels[i]
            if coarse_label != "content":
                successful_list.append("")
                continue

            revision = df_master["used_revision"].tolist()[i]
            context = df_master["used_context"].tolist()[i]

            pred_label = self.predict_successful_sentence_pairs(revision, context)
            successful_list.append(pred_label)

        df_master["successfulness"] = successful_list
        self.master_df = df_master

    def predict_successful_sentence_pairs(self, revision, context):

        test_token = self.tokenizer_success(revision, context, padding="max_length", truncation=True, return_tensors="pt")

        pred_labels = []
        with torch.no_grad():
            test_mask_token = test_token['attention_mask'].to(self.device)
            test_input_ids_tokens = test_token['input_ids'].squeeze(1).to(self.device)
            logit = self.successfulness_classifer_model(test_input_ids_tokens, test_mask_token)
            output = torch.sigmoid(logit)
            predict = output.squeeze().round().cpu().tolist()
            if predict == 0:
                predict = "unsuccessful"
            else:
                predict = "successful"
            return predict

    ################ predict evidence reasoning
    def predict_evidence_reasoning(self):
        print("predicting evidence reasoning...")
        df = self.master_df
        res_list = []
        for i in tqdm(range(len(df))):

            successfulness = df["successfulness"].tolist()[i]
            if str(successfulness) == "":
                res_list.append("")
                continue

            first_sentence = df['old_sentence'].iloc[i]
            second_sentence = df['new_sentence'].iloc[i]
            article = self.get_article(self.topic)

            if str(second_sentence) != "":
                sentence = second_sentence
            else:
                sentence = first_sentence

            context = f'''
                Read an article
                {article}

                You need to identify the giving sentence is an evidence sentence or reasoning sentence.

                Your output should choose from the list [evidence, reasoning]
                The output is: <<<Your output here>>>
                '''

            input = f'''
                This is the given sentence: {sentence}
                '''

            try:
                completion = client.chat.completions.create(
                    model="gpt-35-turbo",  # model = "gpt-4, gpt-35-turbo"
                    temperature=0,
                    messages=[{"role": "system", "content": context},
                              {"role": "user", "content": input}])

                completion = completion.choices[0].message.content
                res = self.extract_result(completion)
                res_list.append(res)

            except Exception as err:
                res_list.append("")
                time.sleep(3)
                print(err)

        df["fine_label"] = res_list

        self.master_df = df

    def extract_result(self, completion):
        if "evidence" in completion:
            res = "evidence"
        else:
            res = "reasoning"
        return res.strip()

    def get_article(self, type="mvp"):
        if type == "mvp":
            text = '''
            A Brighter Future

            Hannah Sachs

            The unpaved dirt road made our car jump as we traveled to the Millennium Village in Sauri (sah-ooh-ree), Kenya. We passed the market where women sat on the dusty ground selling bananas. Little kids were wrapped in cloth on their mothers' backs, or running around in bare feet and tattered clothing. When we reached the village, we walked to the Bar Sauri Primary School to meet the people. Welcoming music and singing had almost everyone dancing. We joined the dancing and clapped along to the joyful, lively music.

            The year was 2010 , the first time I had ever been to Sauri. With the help of the Millennium Villages project, the place would change dramatically in the coming years. The Millennium Villages project was created to help reach the Millennium Development Goals.

            The plan is to get people out of poverty, assure them access to health care and help them stabilize the economy and quality of life in their communities. Villages get technical advice and practical items, such as fertilizer, medicine and school supplies. Local leaders take it from there. The goals are supposed to be met by 2025; some other targets are set for 2035. We are halfway to 2025 , and the world is capable of meeting these goals. But our first glimpse of Sauri showed us that there was plenty of work to do. 

            The Fight for Better Health

            On that day in 2010, we followed the village leaders into Yala Sub-District Hospital. It was not in good shape. There were three kids to a bed and two adults to a bed. The rooms were packed with patients who probably would not receive treatment, either because the hospital did not have it or the patients could not afford it. There was no doctor, only a clinical officer running the hospital. There was no running water or electricity. It is hard for me to see people sick with preventable diseases, people who are near death when they shouldn't have to be. I just get scared and sad.

            Malaria (mah-lair-eeh-ah) is one disease, common in Africa, that is preventable and treatable. Mosquitoes carry malaria, and infect people by biting them. Kids can die from it easily, and adults get very sick.

            Mosquitoes that carry malaria come at night. A bed net, treated with chemicals that last for five years, keeps malarial mosquitoes away from sleeping people. Each net costs $5. There are some cheap medicines to get rid of malaria too. The solutions are simple, yet 20,000 kids die from the disease each day. So sad, and so illogical. Bed nets could save millions of lives.

            Water, Fertilizer, Knowledge

            We walked over to see the farmers. Their crops were dying because they could not afford the necessary fertilizer and irrigation. Time and again, a family will plant seeds only to have an outcome of poor crops because of lack of fertilizer and water. Each year, the farmers worry: Will they harvest enough food to feed the whole family? Will their kids go hungry and become sick?

            Many kids in Sauri did not attend school because their parents could not afford school fees. Some kids are needed to help with chores, such as fetching water and wood. In 2010, the schools had minimal supplies like books, paper and pencils, but the students wanted to learn. All of them worked hard with the few supplies they had. It was hard for them to concentrate, though, as there was no midday meal. By the end of the day, kids didn't have any energy.



            A Better Life-2018

            The people of Sauri have made amazing progress in just eight years. The Yala Sub-District Hospital has medicine, free of charge, for all of the most common diseases. Water is connected to the hospital, which also has a generator for electricity. Bed nets are used in every sleeping site in Sauri. The hunger crisis has been addressed with fertilizer and seeds, as well as the tools needed to maintain the food supply. There are no school fees, and the school now serves lunch for the students. The attendance rate is way up.

            Dramatic changes have occurred in 80 villages across sub-Saharan Africa. The progress is encouraging to supporters of the Millennium Villages project. There are many solutions to the problems that keep people impoverished. What it will really take is for the world to work together to change poverty-stricken areas for good. When my kids are my age, I want this kind of poverty to be a thing of history. It will not be an easy task. But Sauri's progress shows us all that winning the fight against poverty is achievable in our lifetime.
            '''
        else:
            text = '''
            The importance of space exploration
            A Question to Consider

            Is space exploration really desirable when so much needs to be done on Earth? This is a question that has been asked for several decades and requires serious consideration.

            The arguments against space exploration stem from a belief that the money spent could be used differently - to improve people's lives. In 1953, President Eisenhower captured this viewpoint. He opposed the space program, saying that each rocket fired was a theft from citizens that suffered from hunger and poverty.

            Indeed, over 46.2 million Americans (15%) live in poverty. Nearly half of all Americans also have difficulty paying for housing, food, and medicine at some point in their lives. In other countries, people are dying because they do not have access to clean water, medical care, or simple solutions that prevent the spread of diseases. For example, malaria, a disease spread by mosquito bites, kills many people in Africa every year. It is possible to lower the spread of this disease by hanging large nets over beds that protect people from being bitten as they sleep. These nets cost only $5; however, most people affected by malaria cannot afford these nets.

            It is not just people that need help. The Earth is suffering also. Many scientists believe that pollution from burning fossil fuels (gasoline and oil) is harming our air and oceans. We need new, cleaner forms of energy to power cars, homes, and factories. A program to develop clean energy could be viewed as a worthy investment.

            Maybe exploring space should not be a priority when there is so much that needs to be done on Earth. Right now, the government spends 19 billion dollars a year for space exploration. Some people think that this money should be spent instead to help heal the people and the Earth.

            Tangible Benefits of Space Exploration

            People in favor of space exploration argue that 19 billion dollars is not too much. It is only 1.2% of the total national budget. Compare this to the 670 billion dollars the US spends for national defense (26.3% of the national budget), or the 70 billion dollars spent on education (4.8% of the budget), or the 6.3 billion dollars spent on renewable (clean) energy.

            The investment in space exploration is especially worthwhile because it has led to many tangible benefits, for example, in the area of medicine. Before NASA allowed astronauts to go on missions, scientists had to find ways to monitor their health under stressful conditions. This was to ensure the safety of the astronauts under harsh conditions, like those they would experience on launch and return. In doing this, medical instruments were developed and doctors learned about the human body's reaction to stress.

            In rising to meet the challenges of space exploration, NASA scientists have developed other innovations that have improved our lives. These include better exercise machines, better airplanes, and better weather forecasting. All these resulted from technologies that NASA engineers developed to make space travel possible.

            Even the problems of hunger and poverty can be tackled by space exploration. Satellites that circle Earth can monitor lots of land at once. They can track and measure the condition of crops, soil, rainfall, drought, etc. People on Earth can use this information to improve the way we produce and distribute food. So, when we fund space exploration, we are also helping to solve some serious problems on Earth.

            The Spirit of Exploration

            Beyond providing us with inventions, space exploration is important for the challenge it provides and the motivation to bring out the best in ourselves. Space exploration helps us remain a creative society. It makes us strive for better technologies and more scientific knowledge. Often, we make progress in solving difficult problems by first setting challenging goals, which inspire innovative work.

            Finally, space exploration is important because it can motivate beneficial competition among nations. Imagine how much human suffering can be avoided if nations competed with planet-exploring spaceships instead of bomb-dropping airplanes. We saw an example of this in the 1960's. During what is called the Cold War, the United States and Russia competed to prove their greatness in a race to explore space. They each wanted to be the first to land a spacecraft on the moon and visit other planets. This was achieved. It also resulted in many of the technologies and advancements already mentioned. In addition, the 'space race' led to significant investment and progress in American education, especially in math and science. This shows that by looking outward into space, we have also improved life here on Earth.

            Returning to the Question

            All this brings us back to the question: Should we explore space when there is so much that needs to be done on Earth? It is true that we have many serious problems to deal with on Earth, but space exploration is not at odds with solving human problems. In fact, it may even help find solutions. Space exploration will lead to long-term benefits to society that more than justify the immediate cost.'''
        return text

    ################ predict relevance revision

    def predict_topic_relevance(self):
        print("predict_topic_relevance")
        df = self.master_df
        successful_list = df["successfulness"].tolist()
        old_sents = df["old_sentence"].tolist()
        new_sents = df["new_sentence"].tolist()
        keyword_list = []
        for i in tqdm(range(len(df))):
            if successful_list[i] not in ["successful", "unsuccessful"]:
                keyword_list.append("")
                continue
            if str(new_sents[i]) != "":
                sent = new_sents[i]
            else:
                sent = old_sents[i]

            sent = self.preprocess_essay(sent)
            sent = " ".join(sent)
            if len(sent) == 0:
                keyword_list.append("")
                continue

            sent_score = ScoreEssay(sent, topic=self.topic, threshold=0.98)
            npe, added_topic = sent_score.get_npe()
            if npe > 0:
                mentioned_topics = [list(pair.keys())[0] for pair in added_topic.values()]
                added_topic = ",".join(mentioned_topics)
            else:
                added_topic = ""
            keyword_list.append(added_topic)
        df["mentioned_topic"] = keyword_list
        self.master_df = df

    ########## predict revision level
    def get_predict_level(self, old_evidence_level, old_npe, new_npe, old_spc, new_spc):
        print("predict revision levels")
        df_master = self.master_df
        first_npe = old_npe
        second_npe = new_npe
        first_spc = old_spc
        second_spc = new_spc
        first_evidence_level = old_evidence_level
        first_action = df_master["old_sentence_aligned_id"].tolist()
        coarse_labels = df_master["coarse_label"].tolist()
        fine_labels = df_master["fine_label"].tolist()
        successful_labels = df_master["successfulness"].tolist()


        if first_evidence_level == 1:
            if (("content" not in coarse_labels) and ("surface" not in coarse_labels)) or (first_action.count("DELETE")==len(first_action)):
                level = 1.0  # no revision in content or surface or all deletion
            elif "content" not in coarse_labels:
                level = 1.1  # surface revision
            elif first_npe == second_npe:
                keywords = df_master[df_master["successfulness"]=="unsuccessful"]["mentioned_topic"].tolist()
                if len(keywords) == keywords.count(""):
                    level = 1.3 # Repeated evidence from first draft
                else:
                    level = 1.2 # Added evidence but not text based
            elif first_npe > second_npe:
                level = 1.4 # delete evidence but still very vague or general
            else:
                keywords = df_master[df_master["successfulness"] == "unsuccessful"]["mentioned_topic"].tolist()
                if len(keywords) > keywords.count(""):
                    level = 1.5 # Added evidence but still very vague or general
                else:
                    level = 1.3

        elif first_evidence_level == 2:
            if (("content" not in coarse_labels) and ("surface" not in coarse_labels)) or (first_action.count("DELETE")==len(first_action)):
                level = 2.0  # no revision in content or surface or all deletion
            elif "content" not in coarse_labels:
                level = 2.1  # surface revision
            elif first_npe < second_npe and abs(first_spc - second_spc)<=2:
                level = 2.2 # evidence but not specific enough
            elif abs(first_spc - second_spc)<=2:
                keywords = df_master[df_master["successfulness"] == "unsuccessful"]["mentioned_topic"].tolist()
                if len(keywords) > keywords.count(""):
                    level = 2.3 # Added more details but not text based
                else:
                    level = 2.2
            else: # Successfully added specific details; move to next level
                level = 2.4


        elif first_evidence_level == 3:
            evidence_flag, reasoning_flag = self.get_reasoning_evidence_successfulness(fine_labels, successful_labels)

            if (("content" not in coarse_labels) and ("surface" not in coarse_labels)) or (first_action.count("DELETE")==len(first_action)):
                level = 3.0  # no revision in content or surface or all deletion
            elif "content" not in coarse_labels:
                level = 3.1  # surface revision
            elif "reasoning" not in fine_labels:
                level = 3.2 # no attempt
            elif reasoning_flag == False:
                level = 3.3 # unsuccessful explanation/ reasoning attempt
            else:
                level = 3.4 # successful explanation/ reasoning attempt

        else:
            raise "Not implemented"

        # df_master["revision_levels"] = level
        # self.master_df = df_master
        return level

    def get_reasoning_evidence_successfulness(self, reason_evidence_list, success_list):
        success_reasoning = 0
        success_evidence = 0
        unsuccess_reasoning = 0
        unsuccess_evidence = 0
        for i in range(len(success_list)):
            success = success_list[i]
            reason_evidence = reason_evidence_list[i]
            if success == "" or reason_evidence == "":
                continue
            elif reason_evidence == "reasoning" and success == "successful":
                success_reasoning += 1
            elif reason_evidence == "reasoning" and success == "unsuccessful":
                unsuccess_reasoning += 1
            elif reason_evidence == "evidence" and success == "successful":
                success_evidence += 1
            elif reason_evidence == "evidence" and success == "unsuccessful":
                unsuccess_evidence += 1
            else:
                raise "error"

        if success_reasoning >= unsuccess_reasoning and unsuccess_reasoning <= 3:
            reasoning_flag = True
        else:
            reasoning_flag = False

        if success_evidence >= unsuccess_evidence and unsuccess_evidence <= 3:
            evidence_flag = True
        else:
            evidence_flag = False
        return evidence_flag, reasoning_flag

class ScoreEssay():
    def __init__(self, essay, topic="MVP", threshold=0.7):
        self.window_size = 8
        self.topic = topic
        self.threshold = threshold

        self.embeddings = GloVe(name='6B', dim=50)
        # self.embeddings = FastText("simple")
        # self.embeddings = BertModel.from_pretrained('bert-base-uncased')
        # self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        # self.sent_embedding = SentenceTransformer('all-mpnet-base-v2')
        self.tokenizer = get_tokenizer("basic_english")
        self.example_data, self.topic_data = self.get_evaluation_data(topic=topic)
        self.essay_data = self.preprocess_essay(essay)
        self.topic_embedding, self.topic_tokens = self.get_topic_embedding()
        self.topic_string = self.get_topic_string()
        self.essay_embedding = self.get_essay_embedding()
        self.essay_embedding_window = self.get_essay_embedding_window()
        self.essay_string_window = self.get_essay_string_window()


    def preprocess_essay(self, text):
        # text = text.translate(str.maketrans('', '', string.punctuation))
        text = text.replace("\n", " ")
        # text = text.replace('"', '')
        # text = text.replace("'", "")
        # text = text.replace("“", "")
        text = " ".join([token for token in text.split(" ")])
        text = text.strip()
        # text = text.lower()
        text = text.split()
        return text

    def get_evaluation_data(self, topic="MVP"):
        base_path = f"{BASE_DIR}/statics/data/{topic}"
        example_path = os.path.join(base_path, "examples.txt")
        topics_path = os.path.join(base_path, "topics.txt")
        example_list = []
        topic_list = []
        with open(example_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                data = line.split(",")
                data = [d.strip().split() for d in data]
                example_list.append(data)

        with open(topics_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                data = line.split(",")
                data = [d.strip() for d in data]
                topic_list.append(data)
        return example_list, topic_list

    # def get_glove_embedding(self):
    #     embeddings_dict = {}
    #     with open(f'{BASE_DIR}/statics/data/glove.6B/glove.6B.300d.txt', 'r') as f:
    #         for line in f:
    #             values = line.split()
    #             word = values[0]
    #             vectors = np.asarray(values[1:], 'float32')
    #             embeddings_dict[word] = vectors
    #     return embeddings_dict

    def get_topic_embedding(self):
        embedding = []
        topics = []
        for topic in self.topic_data:
            tmp = self.embeddings.get_vecs_by_tokens(topic, lower_case_backup=True)
            # tmp_list = []
            # for tok in topic:
            #     # inputs = self.bert_tokenizer(tok, return_tensors="pt", truncation=True, padding=True)
            #     # outputs = self.embeddings(**inputs)
            #     # tmp = outputs.pooler_output.squeeze()
            #     tmp = self.sent_embedding.encode(tok, convert_to_tensor=True)
            #     # tmp = self.get_gpt_embedding(tok)
            #     tmp_list.append(tmp)
            # tmp_list = torch.stack(tmp_list)
            # embedding.append(tmp_list)
            embedding.append(tmp)
            topics.append(topic)
        return embedding, topics


    def get_topic_string(self):
        if self.topic == "MVP":
            topics = ["HOSPITALS", "MALARIA", "FARMING", "SCHOOL"]
        else:
            topics = ["HELP_PEOPLE", "HELP_ENVIRONMENT", "TANGIBLE_BENEFITS", "SPIRIT_OF_EXPLORATION"]
        return topics

    def get_essay_embedding(self):
        embedding = self.embeddings.get_vecs_by_tokens(self.essay_data, lower_case_backup=True)
        # embedding = []
        # for tok in self.essay_data:
        #     # inputs = self.bert_tokenizer(tok, return_tensors="pt", truncation=True, padding=True)
        #     # outputs = self.embeddings(**inputs)
        #     # tmp = outputs.pooler_output.squeeze()
        #     tmp = self.sent_embedding.encode(tok, convert_to_tensor=True)
        #     # tmp = self.get_gpt_embedding(tok)
        #     embedding.append(tmp)
        # embedding = torch.stack(embedding)
        return embedding

    def get_similarity(self, embedding1, embedding2):
        cosine = torch.nn.CosineSimilarity(dim=0)
        # embedding1 = torch.norm(embedding1, p=2, dim=1)
        # embedding2 = torch.norm(embedding2, p=2, dim=1)
        cosine_score = cosine(embedding1, embedding2).item()
        return cosine_score

    def get_essay_embedding_window(self):
        window = []
        end = len(self.essay_embedding)
        for i in range(0, end - self.window_size, self.window_size):
        # for i in range(0, end - self.window_size):
            j = i + self.window_size if i + self.window_size < end else end
            trunk = self.essay_embedding[i:j,:]
            window.append(trunk)
        return window

    def get_essay_string_window(self):
        window = []
        end = len(self.essay_data)
        for i in range(0, end - self.window_size, self.window_size):
        # for i in range(0, end - self.window_size):
            j = i + self.window_size if i + self.window_size < end else end
            trunk = self.essay_data[i:j]
            window.append(trunk)
        return window

    def get_npe(self):
        # number of topics mentioned
        string_topic_mapper = {}
        for i in range(len(self.topic_embedding)):
            topics = self.topic_embedding[i]
            # look_this_topic = True
            for k in range(len(topics)):
                token_topic = topics[k]
                string_token_topic = self.topic_tokens[i][k]
                # if look_this_topic is False:
                #     break
                for j in range(len(self.essay_embedding_window)):
                    essay_trunk = self.essay_embedding_window[j]
                    matched_num = 0
                    for l in range(len(essay_trunk)):
                        token_essay = essay_trunk[l]
                        cosine_score = self.get_similarity(token_essay, token_topic)
                        # print(cosine_score)
                        # print(cosine_score, string_token_topic, self.essay_string_window[j][l])
                        if cosine_score > self.threshold:
                            print(cosine_score, string_token_topic, self.essay_string_window[j][l])
                            matched_num += 1
                            if matched_num >= 1:
                                break
                    if matched_num >=1:
                        text_string = " ".join(self.essay_string_window[j])
                        string_topic_mapper[text_string] = {self.topic_string[i]: string_token_topic}

                        print("match", string_token_topic, " ".join(self.essay_string_window[j]), "topic", i)

                        # look_this_topic = False
                        # break
        mentioned_topics = [list(pair.keys())[0] for pair in string_topic_mapper.values()]
        num_topic = len(set(mentioned_topics))
        # print(string_topic_mapper)
        return num_topic, string_topic_mapper

    def get_spc(self):
        # number of examples mentioned
        match_examples = []
        for _ in range(len(self.example_data)):
            match_examples.append([])
        match_count = [0]*len(self.example_data)
        for essay_trunk in self.essay_string_window:
            for i in range(len(self.example_data)):
                example_list = self.example_data[i]
                for j in range(len(example_list)):
                    example_trunk = example_list[j]
                    if len(set(example_trunk) & set(essay_trunk)) >= 2:
                        match_count[i] += 1
                        match_examples[i].append(example_trunk)
                        # print(" ".join(essay_trunk), ",", " ".join(example_trunk))
                        break

        return match_count, match_examples


    def get_feedback_level(self):

        npe, input_topic_mapper = self.get_npe()
        spc_scores, match_examples = self.get_spc()
        spc = sum(spc_scores)

        if (spc <= 5 and spc >=3) and npe >= 2:
            level = [2,3]
        elif spc >5 and npe >=3:
            level = [3,4]
        else:
            level = [1,2]
        return level, npe, spc, match_examples, input_topic_mapper

    def get_feedback_level_v2(self):

        npe, input_topic_mapper = self.get_npe()
        spc_scores, match_examples = self.get_spc()
        spc = sum(spc_scores)

        if npe >= 3 and spc >=5:
            level = 3
        elif npe >=3 and spc < 5:
            level = 2
        else:
            level = 1
        # if (spc <= 5 and spc >= 3) and npe >= 2:
        #     level = [2, 3]
        # elif spc > 5 and npe >= 3:
        #     level = [3, 4]
        # else:
        #     level = [1, 2]
        return level, npe, spc, match_examples, input_topic_mapper
