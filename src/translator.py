import os
from openai import AzureOpenAI
from dotenv import load_dotenv


load_dotenv()
client = AzureOpenAI(
    api_key=os.getenv("API_KEY"),
    api_version=os.getenv("API_VERSION"),
    azure_endpoint=os.getenv("AZURE_ENDPOINT")
)


def get_translation(post: str) -> str:
    context = "Translate the query into English. If you can't recognize the language or if the query doesn't make sense, just respond 'Unintelligible or malformed text.'"
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": context
            },
            {
                "role": "user",
                "content": post
            }
        ]
    )
    return response.choices[0].message.content

def get_language(post: str) -> str:
    context =  "Tell me the language that the query is in. If you can't translate it into English, just respond 'Unintelligible or malformed text.'"
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": context
            },
            {
                "role": "user",
                "content": post
            }
        ]
    )
    return response.choices[0].message.content


def translate_content(post: str) -> tuple[bool, str]:
    try:
        language = get_language(post)
        if not isinstance(language, str) or not language.strip():
            raise ValueError("Unintelligible or malformed text.")
        if language.lower() == "english":
            return (True, post)
        else:
            translation = get_translation(post)

            if not isinstance(translation, str) or not translation.strip():
                raise ValueError("Unintelligible or malformed text.")
            return (False, translation)
        
    except Exception as error:
        print(f"Error processing the post: {error}")
        return (False, "Unintelligible or malformed text.")