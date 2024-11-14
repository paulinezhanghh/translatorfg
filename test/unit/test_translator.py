from src.translator import translate_content
import src.translator
from openai import AzureOpenAI
from difflib import SequenceMatcher
from sentence_transformers import util, SentenceTransformer
from unittest.mock import patch
from dotenv import load_dotenv
import os

model = SentenceTransformer('all-MiniLM-L6-v2')

load_dotenv()
client = AzureOpenAI(
    api_key=os.getenv("API_KEY"),
    api_version=os.getenv("API_VERSION"),
    azure_endpoint=os.getenv("AZURE_ENDPOINT")
)
def eval_single_response_translation(expected_answer: str, llm_response: str) -> float:
    return float(util.pytorch_cos_sim(model.encode(expected_answer), model.encode(llm_response))[0][0])

def eval_single_response_complete(expected_answer: tuple[bool, str], llm_response: tuple[bool, str]) -> float:
    score = 0
    if expected_answer[0] == llm_response[0]:
        score += 0.2
    sim_score = eval_single_response_translation(expected_answer[1], llm_response[1]) * 0.8
    score += sim_score
    return min(1.0, score)


# -----------------------
# Unit Tests
# -----------------------
def test_llm_gibberish_response1():
    is_english, translated_content = translate_content("wdejkcgvk")
    assert is_english == False
    assert translated_content == "Unintelligible or malformed text."

def test_llm_gibberish_response2():
    is_english, translated_content = translate_content("...")
    assert is_english == False
    assert translated_content == "Unintelligible or malformed text."

# This tests will pass in final deliverables
# def test_llm_normal_response_chinese():
#     is_english, translated_content = translate_content("这是一条中文消息")
#     assert eval_single_response_complete((False, "This is a Chinese message."), (is_english, translated_content)) >= 0.9

# def test_llm_normal_response_french():
#     is_english, translated_content = translate_content("Ceci est un message en français.")
#     assert eval_single_response_complete((False, "This is a message in French."), (is_english, translated_content)) >= 0.9

def test_llm_normal_response_spanish():
    is_english, translated_content = translate_content("Este es un mensaje en español.")
    assert eval_single_response_complete((False, "This is a message in Spanish."), (is_english, translated_content)) >= 0.9

# -----------------------
# Mock Tests
# -----------------------
@patch.object(src.translator.client.chat.completions, 'create')
def test_llm_gibberish__response1(mocker):
    mocker.return_value.choices[0].message.content = "Unintelligible or malformed text."


    is_english, translated_content = translate_content("dhshghsdgashj.")

    assert is_english == False
    assert translated_content == "Unintelligible or malformed text."



@patch.object(src.translator.client.chat.completions, 'create')
def test_llm_normal_response_japanese(mocker):
    mocker.return_value.choices[0].message.content = "I walked up some stairs."


    is_english, translated_content = translate_content("私は階段を上った。")
    similarity_score = eval_single_response_translation("I walked up some stairs.", translated_content)

    assert is_english == False
    assert similarity_score >= 0.9, f"Similarity is actually {similarity_score}"


@patch.object(src.translator.client.chat.completions, 'create')
def test_llm_normal_response_spanish(mocker):
    mocker.return_value.choices[0].message.content = "This is a message in Spanish."

    is_english, translated_content = translate_content("Este es un mensaje en español.")
    similarity_score = eval_single_response_translation("This is a message in Spanish.", translated_content)

    assert is_english == False
    assert similarity_score >= 0.9, f"Similarity is actually {similarity_score}"