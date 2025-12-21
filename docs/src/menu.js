<nav>
        <h3>目次</h3>
        <ul id="toc"></ul> 
</nav>

<script>
const toc = document.getElementById('toc');
const headings = document.querySelectorAll('h2');

headings.forEach((heading, i) => {
    if (!heading.id) heading.id = `heading-${i}`;
    const link = document.createElement('a');
    link.href = `#${heading.id}`;
    link.textContent = heading.textContent;
    const li = document.createElement('li');
    li.appendChild(link);
    toc.appendChild(li);
});
</script>