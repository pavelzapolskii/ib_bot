#!/usr/bin/env python3
"""Get Telegram chat ID by fetching recent updates."""

import requests
import os
from dotenv import load_dotenv

load_dotenv()

TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '8370064054:AAH0g71KvuN_EcsviEohYTstQTRe54XVljs')

def get_updates():
    url = f"https://api.telegram.org/bot{TOKEN}/getUpdates"
    response = requests.get(url)
    data = response.json()

    if not data.get('ok'):
        print(f"Error: {data}")
        return

    updates = data.get('result', [])

    if not updates:
        print("No messages found.")
        print("Please send a message to your bot @ib_adviser_bot first!")
        return

    print("Found chats:")
    print("-" * 50)

    seen_chats = set()
    for update in updates:
        message = update.get('message', {})
        chat = message.get('chat', {})
        chat_id = chat.get('id')

        if chat_id and chat_id not in seen_chats:
            seen_chats.add(chat_id)
            chat_type = chat.get('type', 'unknown')

            if chat_type == 'private':
                name = f"{chat.get('first_name', '')} {chat.get('last_name', '')}".strip()
                username = chat.get('username', '')
                print(f"Private chat with {name} (@{username})")
            elif chat_type in ['group', 'supergroup']:
                name = chat.get('title', 'Unknown Group')
                print(f"Group: {name}")

            print(f"  Chat ID: {chat_id}")
            print()

    if seen_chats:
        print("-" * 50)
        print("Copy the Chat ID and add it to your .env file as TELEGRAM_CHAT_ID")

if __name__ == '__main__':
    get_updates()
