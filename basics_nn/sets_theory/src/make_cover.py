import asyncio
from playwright.async_api import async_playwright
import os

async def html_to_jpeg():
    async with async_playwright() as p:
        # ブラウザをバックグラウンドで起動
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        
        # HTMLファイルの絶対パスを取得
        html_path = os.path.abspath("cover.html")
        
        # HTMLファイルをブラウザで開く
        await page.goto(f"file://{html_path}")
        
        # HTML内で指定したサイズ（1414x2000）にビューポートを設定
        await page.set_viewport_size({"width": 1414, "height": 2000})
        
        # 完全に読み込まれるまでわずかに待機（画像描画の安定化）
        await page.wait_for_timeout(1000)
        
        # JPEG形式で高画質（品質100%）スクリーンショットを保存
        await page.screenshot(
            path="kindle_cover.jpg",
            type="jpeg",
            quality=100,
            full_page=True
        )
        
        await browser.close()
        print("成功: kindle_cover.jpg を出力しました！")

# 実行
if __name__ == "__main__":
    asyncio.run(html_to_jpeg())