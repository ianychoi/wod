#-*- coding: utf-8 -*-
#import boto3
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from tempfile import mkdtemp
from webdriver_manager.chrome import ChromeDriverManager

import os
import random
import re
import sys
import time


def _build_driver():
    """Create the headless Chrome WebDriver used inside the Lambda container."""
    options = webdriver.ChromeOptions()
    service = webdriver.ChromeService("/opt/chromedriver")

    options.binary_location = '/opt/chrome/chrome'
    options.add_argument("--headless")
    options.add_argument('--no-sandbox')
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--single-process")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-dev-tools")
    options.add_argument("--no-zygote")
    options.add_argument(f"--user-data-dir={mkdtemp()}")
    options.add_argument(f"--data-path={mkdtemp()}")
    options.add_argument(f"--disk-cache-dir={mkdtemp()}")
    options.add_argument("--remote-debugging-port=9222")

    driver = webdriver.Chrome(options=options, service=service)
    driver.implicitly_wait(60)
    return driver


def crawl_posts(urls, output_dir=None, driver=None):
    """Crawl WOD post URLs and return structured post dicts.

    This is the reusable core behind the Lambda ``handler`` and the MCP
    ``ingest_data`` tool. Each returned dict has the same shape as an entry in
    llmops/sample.json::

        {post_id, actor_id, name, content, post_date, photos:[filenames]}

    Screenshots and photos are still saved to ``output_dir`` (defaults to the
    OUTPUT_PATH_POST_SCREENSHOTS env var) so existing behaviour is preserved;
    the difference from the old handler is that this returns data instead of
    only a status code. ``content``/``post_date``/``name``/``actor_id`` are best
    effort from the page and may be empty when not scrapable.
    """
    output_dir = output_dir or os.getenv('OUTPUT_PATH_POST_SCREENSHOTS', '.')
    owns_driver = driver is None
    if owns_driver:
        driver = _build_driver()

    posts = []
    try:
        for url in urls:
            url = url.strip()
            if not url:
                continue

            pattern = r"/groups/(\d+)/posts/(\d+)/"
            match = re.search(pattern, url)
            if not match:
                print("No match found for URL:", url)
                continue
            group_id = match.group(1)
            post_id = match.group(2)

            driver.get(url)
            time.sleep(random.uniform(int(os.getenv('WAIT_POST_LOAD_MIN', 10)),
                                      int(os.getenv('WAIT_POST_LOAD_MAX', 20))))
            webdriver.ActionChains(driver).send_keys(Keys.ESCAPE).perform()
            driver.save_screenshot(output_dir + '/post-' + post_id + '.png')

            soup = BeautifulSoup(driver.page_source, 'html.parser')
            a_tags = soup.find_all('a', href=lambda x: x and '/photo/' in x)

            photos = []
            photo_count = 0
            for a_tag in a_tags:
                img_tag = a_tag.find('img')
                if img_tag:
                    photo_url = img_tag['src']
                    filename = "photo-%s-%s.jpg" % (post_id, photo_count + 1)
                    response = requests.get(photo_url)
                    with open(output_dir + "/" + filename, 'wb') as file:
                        file.write(response.content)
                    photos.append(filename)
                    photo_count += 1

            posts.append({
                "post_id": post_id,
                "group_id": group_id,
                "actor_id": "",
                "name": "",
                "content": "",
                "post_date": "",
                "photos": photos,
            })
            print("Downloaded", photo_count, "photos from post", post_id)
    finally:
        if owns_driver:
            driver.quit()

    return posts


def handler(event, context):

    load_dotenv(verbose=True)

    options = webdriver.ChromeOptions()
    service = webdriver.ChromeService("/opt/chromedriver")

    options.binary_location = '/opt/chrome/chrome'
    options.add_argument("--headless")
    options.add_argument('--no-sandbox')
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--single-process")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-dev-tools")
    options.add_argument("--no-zygote")
    options.add_argument(f"--user-data-dir={mkdtemp()}")
    options.add_argument(f"--data-path={mkdtemp()}")
    options.add_argument(f"--disk-cache-dir={mkdtemp()}")
    options.add_argument("--remote-debugging-port=9222")


    # Initialize the WebDriver
    driver = webdriver.Chrome(options=options, service=service)
    driver.implicitly_wait(60)

    post_count = 0

    try:

        with open('posts.txt', 'r') as file:
            urls = file.readlines()

        for url in urls:
            url = url.strip()       # Strip leading and trailing whitespace
            if not url:
                continue            # Skip empty lines

            # Step 1: Extract the group ID and post ID from the URL
            pattern = r"/groups/(\d+)/posts/(\d+)/"
            match = re.search(pattern, url)
            if match:
                group_id = match.group(1)
                post_id = match.group(2)
            else:
                print("No match found for URL:", url)
                continue

            # Step 2: Open the URL of the post
            driver.get(url)
            time.sleep(random.uniform(int(os.getenv('WAIT_POST_LOAD_MIN')),
                                    int(os.getenv('WAIT_POST_LOAD_MAX'))))  # Wait for the page to load

            # Step 3: Press the ESC key to close the login popup
            webdriver.ActionChains(driver).send_keys(Keys.ESCAPE).perform()

            # Step 4: Take a screenshot of the post
            driver.save_screenshot(os.getenv('OUTPUT_PATH_POST_SCREENSHOTS') + '/post-' + post_id + '.png')

            # Step 5: Retrieve photo URLs from the post
            soup = BeautifulSoup(driver.page_source, 'html.parser')

            # Find the "<a>"s containing the photos
            a_tags = soup.find_all('a', href=lambda x: x and '/photo/' in x)

            photo_count = 0
            for a_tag in a_tags:
                img_tag = a_tag.find('img')
                if img_tag:
                    photo_url = img_tag['src']
                    # print(photo_url)
                    
                    # Download the photo
                    response = requests.get(photo_url)
                    with open(os.getenv('OUTPUT_PATH_POST_SCREENSHOTS') + "/photo-%s-%s.jpg" % (post_id, photo_count+1), 'wb') as file:
                        file.write(response.content)
                    photo_count += 1

            print("Downloaded", photo_count, "photos from post", post_id)
            post_count += 1

        print("Total processed ", post_count, " posts")

    finally:
        # Close the WebDriver
        driver.quit()
        return {'statusCode': 200, 'body': ('Crawling completed: %s' % (post_count)) }
