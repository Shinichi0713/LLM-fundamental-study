const puppeteer = require('puppeteer');

(async () => {
  const browser = await puppeteer.launch();
  const page = await browser.newPage();
  
  // HTMLファイルを読み込む（絶対パス）
  await page.goto('file:///C:/path/to/your/file.html', {waitUntil: 'networkidle0'});
  
  // A4サイズにビューポートを設定
  await page.setViewport({ width: 794, height: 1123, deviceScaleFactor: 2 }); // 96dpi換算のA4サイズ

  await page.screenshot({
    path: 'cover.png',
    fullPage: false,
    clip: { x: 0, y: 0, width: 794, height: 1123 }
  });

  await browser.close();
})();