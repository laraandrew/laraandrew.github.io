# Andrew Lara Blog (`laraandrew.github.io`)

Personal blog built with Jekyll and hosted on GitHub Pages.

- Live site: [blog.andrewlara.com](https://blog.andrewlara.com)
- GitHub Pages fallback: [laraandrew.github.io](https://laraandrew.github.io)

## Repo Structure

- `_posts/` - Blog posts (`YYYY-MM-DD-slug.md`)
- `assets/` - Styles and static assets
- `_config.yml` - Jekyll site configuration
- `CNAME` - Custom domain mapping for GitHub Pages

## Writing a New Post

1. Create a file in `_posts/` named `YYYY-MM-DD-post-slug.md`.
2. Add front matter at the top:

```markdown
---
title: "Post Title"
date: YYYY-MM-DD
categories: [engineering]
tags: [ai, notes]
---
```

3. Write the post in Markdown.
4. Commit and open a PR to `main`.
5. Merge to publish (GitHub Pages deploys automatically).

## Local Preview

```bash
bundle install
bundle exec jekyll serve
```

Then open `http://127.0.0.1:4000`.

## Notes

- If custom-domain HTTPS is still provisioning, `laraandrew.github.io` may be the more reliable preview URL temporarily.
- For contributor automation guidance, see [AGENT_GUIDE.md](AGENT_GUIDE.md).
