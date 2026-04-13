# Agent Guide — laraandrew.github.io

This is a Jekyll blog hosted on GitHub Pages. Posts are plain Markdown files.
No build step required — GitHub Pages builds automatically on every push to `main`.

---

## How to add a new blog post

1. Create a file in `_posts/` named: `YYYY-MM-DD-post-slug.md`
2. Add frontmatter at the top:

   ```markdown
   ---
   title: "Your Post Title"
   date: YYYY-MM-DD
   categories: [engineering]
   tags: [ai, backend, etc]
   ---
   ```

3. Write content below the frontmatter in standard Markdown.
4. Commit and push to `main` — live in ~30 seconds.

---

## Categories in use

- `engineering`
- `ai`
- `projects`
- `meta`

---

## Rules

- Filename must be `YYYY-MM-DD-slug.md` — no spaces, use hyphens.
- The `date:` in frontmatter must match the date in the filename.
- Images go in `assets/img/posts/YYYY-MM-DD/` and reference as `/assets/img/posts/YYYY-MM-DD/image.jpg`.
- Do **not** edit `_config.yml` or `CNAME` without checking with Andrew.

---

## Key files

| File | Purpose |
|------|---------|
| `_config.yml` | Site title, URL, theme settings |
| `CNAME` | Custom domain (`blog.andrewlara.com`) |
| `Gemfile` | Jekyll gem dependencies |
| `_posts/` | All blog posts |
| `assets/img/posts/` | Post images |

---

## Deployment

- **No manual deploy needed.** Push to `main` = published.
- GitHub Pages uses the `jekyll-theme-chirpy` theme.
- Live at: `https://blog.andrewlara.com` (and `https://laraandrew.github.io`)
