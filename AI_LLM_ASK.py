#this part is a API tools for the online LLM models
#which will be used to generate the answer for the users
#which needs the user to get their own API key
from http import HTTPStatus;
import dashscope;
import os;
def call(message:list,apikey="",model="qwen-plus"):
    os.environ["DASHSCOPE_API_KEY"]=apikey;
    dashscope.api_key=apikey;
    response=dashscope.Generation.call(
        model=model,
        messages=message,
        result_fromat="message",
    )
    if(response.status_code==HTTPStatus.OK):
        print(response);
        return response;
    else:
        print(f"Request id : {response.request_id}, "
              f"Status code: {response.status_code}, "
              f"Error code: {response.code}"
              f"Error message: {response.messgae}");
        return None;

