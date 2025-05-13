# 3d_building_plan_generator.py

import os
from langchain_openai import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from dotenv import load_dotenv

from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
import openai
import webbrowser

# Step 1: Set API key
load_dotenv()
# Step 2: Use LangChain + GPT-4 to generate building plan
llm = ChatOpenAI(model="gpt-4", temperature=0.7)

# Prompt template
prompt = ChatPromptTemplate.from_template("""
You are an architect AI. Generate a 3D-style building plan description based on the following request:
- Type: {building_type}
- Area (in sq ft): {area}
- Floors: {floors}
- Rooms: {rooms}
- Style: {style}

Describe the floor plan, materials, design elements, and 3D layout in great visual detail suitable for image generation.
""")

# LangChain chain
plan_chain = LLMChain(llm=llm, prompt=prompt)

# Step 3: Function to generate image with DALL·E
def generate_image(prompt_text):
    print("🎨 Generating image with DALL·E...")
    response = openai.images.generate(
        model="dall-e-3",
        prompt=prompt_text,
        n=1,
        size="1024x1024"
    )
    image_url = response.data[0].url
    print("🖼️ Image URL:", image_url)
    webbrowser.open(image_url)
    return image_url

# Step 4: Main function to combine everything
def main():
    print("🏗️ 3D Building Plan Generator")
    
    building_type = input("🏢 Building type (e.g., Residential, Office, Villa): ")
    area = input("📏 Area (in sq ft): ")
    floors = input("🏬 Number of floors: ")
    rooms = input("🛏️ Rooms needed (e.g., 3 BHK + 1 Study): ")
    style = input("🎨 Style (e.g., Modern, Traditional, Minimalist): ")
    
    description = plan_chain.run({
        "building_type": building_type,
        "area": area,
        "floors": floors,
        "rooms": rooms,
        "style": style
    })

    print("\n📝 Generated Building Plan Description:\n")
    print(description)

    generate_image(description)

if __name__ == "__main__":
    main()
