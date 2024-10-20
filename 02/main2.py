import requests
from bs4 import BeautifulSoup

target_url = "https://www.landonhotel.com"

response = requests.get(target_url)

if response.status_code == 200:
    soup = BeautifulSoup(response.content, "html.parser")

    text = ""
    
    for paragraph in soup.find_all("p"):
        text += paragraph.get_text()

    with open('02/website_text.txt', 'w', encoding='utf-8') as text_file:
        text_file.write(text)
    
    print("Text extracted and saved successfully!")

else: 
    print(f"Error: Failed to retreive website content. Status code: {response.status_code}")