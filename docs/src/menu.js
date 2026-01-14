/* 目次全体のコンテナ */
nav {
    background-color: #f8f9fa;
    border: 1px solid #e1e4e8;
    border-radius: 12px;
    padding: 24px;
    margin: 20px 0 40px 0;
    max-width: 600px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.05);
}

/* 「目次」というタイトル */
nav h3 {
    margin-top: 0;
    margin-bottom: 16px;
    padding-bottom: 8px;
    border-bottom: 2px solid #0969da;
    color: #24292f;
    font-size: 1.2rem;
    display: flex;
    align-items: center;
}

/* タイトルの前にアイコン（絵文字）を追加 */
nav h3::before {
    content: "📖";
    margin-right: 8px;
}

/* リストのスタイル調整 */
#toc {
    list-style: none;
    padding-left: 0;
    margin: 0;
}

#toc li {
    margin-bottom: 8px;
    line-height: 1.4;
}

/* リンクのスタイル */
#toc a {
    color: #0969da;
    text-decoration: none;
    font-weight: 500;
    transition: all 0.2s ease;
    display: inline-block;
}

#toc a:hover {
    color: #cf222e;
    transform: translateX(5px); /* ホバー時に少し右に動く */
}

/* h3（小見出し）がある場合のネスト表現（JSの修正も必要） */
.toc-h3 {
    padding-left: 20px;
    font-size: 0.9em;
    opacity: 0.8;
}


/* 記事タイトル (h1) */
h1 {
    font-size: 2rem;
    color: #24292f;
    line-height: 1.3;
    padding: 20px 0;
    margin-bottom: 30px;
    border-bottom: 3px double #e1e4e8; /* 二重線で上品に */
    text-align: center; /* タイトルを中央に寄せて特別感を出す */
}

/* セクション見出し (h2) */
h2 {
    font-size: 1.5rem;
    color: #24292f;
    padding: 0.5rem 1rem;
    margin: 40px 0 20px 0;
    background: linear-gradient(transparent 70%, #e8f0fe 70%); /* 下側に薄い色のアクセント */
    border-left: 6px solid #0969da; /* 目次のテーマカラーと合わせる */
    border-radius: 2px;
    display: flex;
    align-items: center;
}

/* 強調文字 (strong) */
strong {
    font-weight: bold;
    color: #cf222e; /* ホバー時の赤色と合わせて統一感を出す */
    background: linear-gradient(transparent 60%, #fff2cc 60%); /* 黄色のマーカー風 */
    padding: 0 2px;
}

/* 引用のコンテナ */
blockquote {
    position: relative;
    padding: 20px 30px;
    margin: 30px 0;
    background-color: #f6f8fa; /* 目次の背景より少しだけ濃いグレー */
    border-left: 5px solid #d0d7de; /* 落ち着いたグレーの境界線 */
    color: #57606a; /* 文字色は少し薄くして引用らしさを出す */
    font-style: italic;
    border-radius: 0 8px 8px 0;
}

/* 引用符のアイコンを装飾として追加 */
blockquote::before {
    content: "“";
    position: absolute;
    top: -5px;
    left: 10px;
    font-size: 40px;
    color: #d0d7de;
    font-family: serif;
    line-height: 1;
}

/* 読者になるボタンのデザイン */
.btn-subscribe {
    display: inline-block;
    padding: 12px 35px; /* 横幅を広めにとって存在感を出します */
    background-color: #383838; /* お好みの色に変更してください */
    color: #ffffff !important;
    text-decoration: none;
    border-radius: 4px;
    font-weight: bold;
    transition: 0.3s;
}

.btn-subscribe:hover {
    background-color: #555555;
    text-decoration: none;
}

<nav>
        <h3>目次</h3>
        <ul id="toc"></ul> 
</nav>



<h2>最後に</h2>
<p>記事を最後まで読んでくださってありがとうございます。<br />記事の内容を読んで頑張れと思っていただいた読者様、是非、読者になって今後も応援してください。</p>
<iframe src="https://blog.hatena.ne.jp/yoshishinnze/yoshishinnze.hatenablog.com/subscribe/iframe" allowtransparency="true" frameborder="0" scrolling="no" width="150" height="28"></iframe>


<script>
const toc = document.getElementById('toc');
// h2 と h3 の両方を取得
const headings = document.querySelectorAll('h2, h3');

headings.forEach((heading, i) => {
    if (!heading.id) heading.id = `heading-${i}`;
    
    const li = document.createElement('li');
    const link = document.createElement('a');
    link.href = `#${heading.id}`;
    link.textContent = heading.textContent;
    
    // h3 の場合はクラスを付与して字下げする
    if (heading.tagName === 'H3') {
        li.classList.add('toc-h3');
    }
    
    li.appendChild(link);
    toc.appendChild(li);
});

// スムーズスクロールを有効にする
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        document.querySelector(this.getAttribute('href')).scrollIntoView({
            behavior: 'smooth'
        });
    });
});
</script>

