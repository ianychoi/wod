from langchain_community.callbacks.manager import get_openai_callback
from langchain.schema import HumanMessage, SystemMessage
from langchain_aws import ChatBedrockConverse
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from typing import List, Dict
import os
from dotenv import load_dotenv
from PIL import Image
import base64
from io import BytesIO
import time
import json
import sys

load_dotenv()

class AIModelComparator:
    def __init__(self):
        # Initialize OpenAI
        self.openai_chat = ChatOpenAI(
            model=os.getenv("OPENAI_MODEL_NAME"),
            temperature=os.getenv("LANGCHAIN_TEMPERATURE", 0.7),
            max_tokens=1500,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Initialize Azure OpenAI
        self.azure_chat = AzureChatOpenAI(
            deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            openai_api_key=os.getenv("AZURE_OPENAI_KEY"),
            temperature=os.getenv("LANGCHAIN_TEMPERATURE", 0.7)
        )
        
        # Initialize Bedrock Claude
        self.bedrock_chat = ChatBedrockConverse(
            model=os.getenv("AWS_BEDROCK_MODEL"),
            credentials_profile_name=os.getenv("AWS_BEDROCK_PROFILE_NAME"),
            region_name=os.getenv("AWS_BEDROCK_REGION_NAME"),
            temperature= os.getenv("LANGCHAIN_TEMPERATURE", 0.7),
            max_tokens= 1500
        )
        
        # 시스템 프롬프트 설정
        self.system_prompt = SystemMessage(
            content="운동 데이터를 분석하고 피드백을 제공하는 전문 트레이너입니다."
        )

    def encode_image(self, image_path: str) -> str:
        """Encode an image file with base64"""
        with Image.open(image_path) as img:
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            return base64.b64encode(buffered.getvalue()).decode()

    def prepare_messages(self, text: str, image_paths: List[str]) -> List[dict]:
        """Message preparation for each model"""
        messages = [self.system_prompt]
        
        if image_paths:
            image_contents = []
            for path in image_paths:
                base64_image = self.encode_image(path)
                image_contents.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{base64_image}"}
                })
            
            messages.append(HumanMessage(content=[
                {"type": "text", "text": text},
                *image_contents
            ]))
        else:
            messages.append(HumanMessage(content=text))
            
        return messages

    async def analyze_with_model(self, model, text: str, image_paths: List[str], model_name: str) -> Dict:
        """Analyze text and images with a given model"""
        start_time = time.time()
        messages = self.prepare_messages(text, image_paths)
        
        try:
            with get_openai_callback() as cb:
                response = await model.ainvoke(messages)
                end_time = time.time()
                
                return {
                    "model": model_name,
                    "response": response.content,
                    "time_taken": end_time - start_time,
                    "total_tokens": cb.total_tokens,
                    "prompt_tokens": cb.prompt_tokens,
                    "completion_tokens": cb.completion_tokens,
                    "total_cost": cb.total_cost
                }
        except Exception as e:
            return {
                "model": model_name,
                "error": str(e),
                "time_taken": time.time() - start_time
            }

    async def compare_models(self, text: str, image_paths: List[str]) -> Dict[str, Dict]:
        tasks = [
            self.analyze_with_model(self.openai_chat, text, image_paths, "OpenAI"),
            self.analyze_with_model(self.azure_chat, text, image_paths, "Azure OpenAI"),
            self.analyze_with_model(self.bedrock_chat, text, image_paths, "Claude (Bedrock)")
        ]
        
        # Execute all tasks concurrently
        results_list = await asyncio.gather(*tasks)
        
        # Convert list of results to a dictionary
        results = {result["model"]: result for result in results_list}
        return results

async def process_entry(comparator: AIModelComparator, entry: Dict) -> Dict:
    post_id = entry['post_id']
    post_date = entry['post_date']
    actor_id = entry['actor_id']
    name = entry['name']
    content = entry['content']
    photos = entry['photos']

    text = f"""
    포스트 ID: {post_id}
    포스트 일자: {post_date}
    사용자 ID: {actor_id}
    사용자 이름: {name}
    오늘의 운동 기록:
    - {content}
    첨부 이미지 파일과 함께 이 운동 포스트 내역을 분석해주세요.
    결과는 포스트 ID 값, 포스트 일자 값, 사용자 ID 값, 사용자 이름을 포함해서 JSON 형태로 출력해줘.
    또한 해당 포스트 내용이 운동과 관련있는지 여부를 체크해서 is_exercise 필드에 True 또는 False로 표시해줘.
    운동에 대한 요약을 exercise_summary 필드에 담아서 출력해줘.
    운동 요약을 분석해서 운동 시간을 분 (minute) 단위로 예상후 exercise_time 필드에 숫자로 출력해줘.
    그리고 운동 요약을 분석해서 소모한 칼로리를 kcal 단위로 예상해서 exercise_calories 필드에 숫자로 출력해줘.
    그 외 모든 데이터를 JSON 필드에 담아 결과는 완전한 JSON 형태, 마크다운을 사용하지 않고 JSON Text만 출력해줘.
    """

    image_paths = ['sample_photos/' + photo for photo in photos]
    
    try:
        results = await comparator.compare_models(text, image_paths)
        output = {
            "post_id": post_id,
            "analysis": {}
        }
        
        print(f"\n=== Post ID {post_id}: Batch finished ===")

        for model_name, result in results.items():
            if "error" in result:
                print(f"Error in {model_name}: {result['error']}", file=sys.stderr)
                output["analysis"][model_name] = {
                    "error": result["error"],
                    "time_taken": result["time_taken"]
                }
            else:
                try:
                    response_json = json.loads(result["response"])
                    output["analysis"][model_name] = {
                        "result": response_json,
                        "time_taken": result["time_taken"],
                        "total_tokens": result.get("total_tokens"),
                        "total_cost": result.get("total_cost")
                    }
                except json.JSONDecodeError as e:
                    print(f"JSON parsing error in {model_name}: {e}", file=sys.stderr)
                    print(f"Raw response: {result['response']}", file=sys.stderr)
                    output["analysis"][model_name] = {
                        "error": "JSON parsing error",
                        "raw_response": result["response"],
                        "time_taken": result["time_taken"]
                    }
        return output
    except Exception as e:
        error_output = {
            "post_id": post_id,
            "error": str(e)
        }
        print(f"Error processing post {post_id}: {str(e)}", file=sys.stderr)
        return error_output

async def process_batch(comparator: AIModelComparator, entries: List[Dict], output_file):
    """Process a batch of entries concurrently"""
    tasks = [process_entry(comparator, entry) for entry in entries]
    results = await asyncio.gather(*tasks)

    for result in results:
        json.dump(result, output_file, ensure_ascii=False, indent=2)
        output_file.write(',\n')

async def main():
    comparator = AIModelComparator()

    # Include a timestamp in the output filename
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_filename = f"analysis_results_{timestamp}.json"
    
    with open("sample.json", "r", encoding="utf-8") as file:
        data = json.load(file)

    # Open the output file for writing
    with open(output_filename, "w", encoding="utf-8") as output_file:
        # Start the JSON array
        output_file.write("[\n")
        
        # Set the batch size
        BATCH_SIZE = 2
        
        # Process the data in batches
        for i in range(0, len(data), BATCH_SIZE):
            batch = data[i:i + BATCH_SIZE]
            await process_batch(comparator, batch, output_file)
        
        # Remove the last comma and newline with JSON closing bracket
        output_file.seek(output_file.tell() - 2, 0)  # Delete the last comma and newline
        output_file.write("\n]")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
