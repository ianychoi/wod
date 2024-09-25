from dotenv import load_dotenv
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

import os
import time


load_dotenv(verbose=True)

# Set the options for the WebDriver
options = Options()
# Recommended options for the WebDriver
options.add_argument("--headless")  # Headless 모드 설정
options.add_argument("--disable-gpu")  # GPU 비활성화 (Linux에서 필요할 수 있음)
options.add_argument("--window-size=1920,1080")  # 창 크기 설정
options.add_argument("--disable-extensions")
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")
# Custom options for the WebDriver
options.add_argument(os.getenv('CHROMEDRIVER_CUSTOM_OPTIONS'))

chromedriver_path = os.getenv('CHROMEDRIVER_PATH')
service = Service(chromedriver_path)


# Initialize the WebDriver
driver = webdriver.Chrome(options=options,
                          service=service)

try:
    # Step 1: Open WOD - login page
    driver.get(os.getenv('WOD_LOGIN_URL'))  # Replace with the actual URL of the login page
    time.sleep(2)  # Wait for the page to load

    # Step 2: Log in to WOD
    email_elem = driver.find_element(By.ID, "email")  # Locate by ID
    email_elem.send_keys(os.getenv('WOD_LOGIN_USERNAME'))
    password_elem = driver.find_element(By.ID, "pass")  # Locate by ID
    password_elem.send_keys(os.getenv('WOD_LOGIN_PASSWORD'))
    password_elem.send_keys(Keys.RETURN)
    time.sleep(5)  # Wait for login to complete

    # Step 3: Take a screenshot of the homepage after login
    driver.save_screenshot(os.getenv('OUTPUT_PATH_SCREENSHOTS') + '/login_succeeded.png')

    # Step 4: Navigate to the WOD group - posts page
    driver.get(os.getenv('WOD_POSTS_URL'))  # Replace with the actual URL of the group
    time.sleep(5)  # Wait for the page to load

    # Step 5: Take a screenshot of the posts page
    driver.save_screenshot(os.getenv('OUTPUT_PATH_SCREENSHOTS') + '/posts.png')

    # Step 6: Save the HTML source of the group page
    with open(os.getenv('OUTPUT_PATH_HTML') + '/output.html', 'w', encoding='utf-8') as file:
        file.write(driver.page_source)

    # [Work in Progress] Step 7: How to scrool down the page?
    body = driver.find_element(By.TAG_NAME, 'body')
    for _ in range(3):
        body.send_keys(Keys.PAGE_DOWN)
        # Waiting
        time.sleep(1)

finally:
    # Close the WebDriver
    driver.quit()
