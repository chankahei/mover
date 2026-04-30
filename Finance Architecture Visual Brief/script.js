import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';

const root = document.documentElement;
let theme = 'light';

function applyTheme(nextTheme) {
  theme = nextTheme;
  root.setAttribute('data-theme', theme);
  mermaid.initialize({
    startOnLoad: false,
    theme: theme === 'dark' ? 'dark' : 'base',
    themeVariables: {
      background: 'transparent',
      primaryColor: theme === 'dark' ? '#222222' : '#ffffff',
      primaryTextColor: theme === 'dark' ? '#f4f4f2' : '#161616',
      primaryBorderColor: theme === 'dark' ? '#ff4a43' : '#ef241c',
      lineColor: theme === 'dark' ? '#b5b5b1' : '#626262',
      secondaryColor: theme === 'dark' ? '#2d2d2d' : '#f7f7f5',
      tertiaryColor: theme === 'dark' ? '#181818' : '#ffffff',
      fontFamily: 'Satoshi, Inter, sans-serif',
    },
    flowchart: {
      curve: 'basis',
      nodeSpacing: 38,
      rankSpacing: 48,
      padding: 16,
    },
    securityLevel: 'loose',
  });
}

async function renderMermaid() {
  const nodes = document.querySelectorAll('.mermaid');
  nodes.forEach((node) => {
    if (node.dataset.source) {
      node.textContent = node.dataset.source;
    } else {
      node.dataset.source = node.textContent;
    }
    node.removeAttribute('data-processed');
  });
  await mermaid.run({ nodes });
}

applyTheme(theme);
renderMermaid();

const chips = document.querySelectorAll('[data-filter]');
const cards = document.querySelectorAll('.signal-card');

chips.forEach((chip) => {
  chip.addEventListener('click', () => {
    const filter = chip.dataset.filter;
    chips.forEach((item) => item.classList.toggle('active', item === chip));
    cards.forEach((card) => {
      const visible = filter === 'all' || card.dataset.kind === filter;
      card.hidden = !visible;
    });
  });
});
