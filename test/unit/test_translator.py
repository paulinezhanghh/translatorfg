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
        score += 0.5
    sim_score = eval_single_response_translation(expected_answer[1], llm_response[1]) * 0.5
    score += sim_score
    return min(1.0, score)


# -----------------------
# Unit Tests
# -----------------------
def test_llm_normal_response_russian():
    is_english, translated_content = translate_content("Это сообщение на русском языке.")
    assert eval_single_response_complete((False, "This is a message in Russian."), (is_english, translated_content)) >= 0.9

def test_llm_normal_response_korean():
    is_english, translated_content = translate_content("이것은 한국어 메시지입니다.")
    assert eval_single_response_complete((False, "This is a message in Korean."), (is_english, translated_content)) >= 0.9

def test_llm_gibberish_response1():
    is_english, translated_content = translate_content("wdejkcgvk")
    assert is_english == False
    assert translated_content == "Unintelligible or malformed text."

def test_llm_gibberish_response2():
    is_english, translated_content = translate_content("...")
    assert is_english == False
    assert translated_content == "Unintelligible or malformed text."

def test_llm_normal_response_spanish():
    is_english, translated_content = translate_content("Este es un mensaje en español.")
    assert eval_single_response_complete((False, "This is a message in Spanish."), (is_english, translated_content)) >= 0.9

def test_llm_normal_response_french():
    is_english, translated_content = translate_content("Je mange une pomme.")
    assert eval_single_response_complete((False, "I am eating an apple."), (is_english, translated_content)) >= 0.9

def test_llm_normal_response_german():
    is_english, translated_content = translate_content("Das Auto ist rot.")
    assert eval_single_response_complete((False, "The car is red."), (is_english, translated_content)) >= 0.9

def test_llm_normal_response_japanese():
    is_english, translated_content = translate_content("私は本を読んでいます。")
    assert eval_single_response_complete((False, "I am reading a book."), (is_english, translated_content)) >= 0.9

def test_llm_normal_response_chinese():
    is_english, translated_content = translate_content("天气很好。")
    assert eval_single_response_complete((False, "The weather is nice."), (is_english, translated_content)) >= 0.9

def test_llm_normal_response_italian():
    is_english, translated_content = translate_content("Mi piace la musica.")
    assert eval_single_response_complete((False, "I like music."), (is_english, translated_content)) >= 0.9

def test_llm_normal_response_spanish():
    is_english, translated_content = translate_content("El gato duerme.")
    assert eval_single_response_complete((False, "The cat is sleeping."), (is_english, translated_content)) >= 0.9

def test_llm_normal_response_portuguese():
    is_english, translated_content = translate_content("Estou aprendendo a programar.")
    assert eval_single_response_complete((False, "I am learning to program."), (is_english, translated_content)) >= 0.9

def test_llm_normal_response_arabic():
    is_english, translated_content = translate_content("الكتاب على الطاولة.")
    assert eval_single_response_complete((False, "The book is on the table."), (is_english, translated_content)) >= 0.9

def test_llm_normal_response_hindi():
    is_english, translated_content = translate_content("मैं स्कूल जा रहा हूँ।")
    assert eval_single_response_complete((False, "I am going to school."), (is_english, translated_content)) >= 0.9

def test_llm_normal_response_dutch():
    is_english, translated_content = translate_content("Het huis is groot.")
    assert eval_single_response_complete((False, "The house is big."), (is_english, translated_content)) >= 0.9
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

@patch.object(src.translator.client.chat.completions, 'create')
def test_llm_normal_response_french(mocker):
    mocker.return_value.choices[0].message.content = "I am eating an apple."
    is_english, translated_content = translate_content("Je mange une pomme.")
    similarity_score = eval_single_response_translation("I am eating an apple.", translated_content)
    assert is_english == False
    assert similarity_score >= 0.9, f"Similarity is actually {similarity_score}"

@patch.object(src.translator.client.chat.completions, 'create')
def test_llm_normal_response_german(mocker):
    mocker.return_value.choices[0].message.content = "The car is red."
    is_english, translated_content = translate_content("Das Auto ist rot.")
    similarity_score = eval_single_response_translation("The car is red.", translated_content)
    assert is_english == False
    assert similarity_score >= 0.9, f"Similarity is actually {similarity_score}"

@patch.object(src.translator.client.chat.completions, 'create')
def test_llm_normal_response_portuguese(mocker):
    mocker.return_value.choices[0].message.content = "I am learning to program."
    is_english, translated_content = translate_content("Estou aprendendo a programar.")
    similarity_score = eval_single_response_translation("I am learning to program.", translated_content)
    assert is_english == False
    assert similarity_score >= 0.9, f"Similarity is actually {similarity_score}"

@patch.object(src.translator.client.chat.completions, 'create')
def test_llm_normal_response_arabic(mocker):
    mocker.return_value.choices[0].message.content = "The book is on the table."
    is_english, translated_content = translate_content("الكتاب على الطاولة.")
    similarity_score = eval_single_response_translation("The book is on the table.", translated_content)
    assert is_english == False
    assert similarity_score >= 0.9, f"Similarity is actually {similarity_score}"

@patch.object(src.translator.client.chat.completions, 'create')
def test_llm_normal_response_hindi(mocker):
    mocker.return_value.choices[0].message.content = "I am going to school."
    is_english, translated_content = translate_content("मैं स्कूल जा रहा हूँ।")
    similarity_score = eval_single_response_translation("I am going to school.", translated_content)
    assert is_english == False
    assert similarity_score >= 0.9, f"Similarity is actually {similarity_score}"

@patch.object(src.translator.client.chat.completions, 'create')
def test_llm_normal_response_dutch(mocker):
    mocker.return_value.choices[0].message.content = "The house is big."
    is_english, translated_content = translate_content("Het huis is groot.")
    similarity_score = eval_single_response_translation("The house is big.", translated_content)
    assert is_english == False
    assert similarity_score >= 0.9, f"Similarity is actually {similarity_score}"

@patch.object(src.translator.client.chat.completions, 'create')
def test_llm_normal_response_chinese(mocker):
    mocker.return_value.choices[0].message.content = "The weather is nice."
    is_english, translated_content = translate_content("天气很好。")
    similarity_score = eval_single_response_translation("The weather is nice.", translated_content)
    assert is_english == False
    assert similarity_score >= 0.9, f"Similarity is actually {similarity_score}"