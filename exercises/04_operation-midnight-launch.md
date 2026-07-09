# 🚀 Operation Midnight Launch

## The NovaGrid Incident

**Time:** ~35 minutes | **Tools:** VS Code, Terminal, Claude Code | **Output:** One glorious `index.html`

---

## ✅ Before You Start

You should already have:

- **VS Code** installed and opening correctly
- **Claude Code** installed (`claude --version` shows a version number) and you've signed in at least once

Don't have these yet? Work through the **[Installation Guide](installation-guide.md)** first — it walks through both, start to finish (~15-20 min).

---

## 📖 The Story

It's 11 PM on a Thursday.

You just got a panicked call from **Mara**, the CEO of **NovaGrid**, a clean-energy analytics startup. Investors are walking through the door at 9 AM sharp for a live demo of the company's new landing page.

Small problem: the freelance developer who was building it? Gone. Vanished. Phone off. LinkedIn deactivated. Left behind nothing but a half-finished `index.html`, a cold cup of coffee, and a sticky note that reads:

> *"good luck 👍"*

You've been promoted (voluntarily? involuntarily? does it matter at this point?) to **Interim Head of Product**. Your mission: ship this landing page before sunrise.

The good news? You have an engineer. A tireless, endlessly patient, slightly literal engineer named **Claude Code**.

The bad news? The board slashed the AI tooling budget last quarter, so you need to be smart about which model you use and how much context you burn through. Also, **Elena** (CTO, currently asleep, terrifying when awake) will inspect your **git history** first thing in the morning. She expects clean branches, meaningful commits, and zero chaos on `main`.

No pressure.

**Let's go.**

---

## 🎯 What You Will Practice

By the end of this exercise, you will have hands-on experience with:

- Setting up and prompting Claude Code effectively
- Writing a structured `CLAUDE.md` to guide your AI engineer
- Using `/model`, `/clear`, `/compact`, and `/context` strategically
- Working with **feature branches**, clean commits, and merges
- Pasting images/screenshots into the terminal as context
- Creating your own **custom slash commands**
- Managing your context window like a pro (because tokens cost money)

---

## 🛠️ Phase 0: Set Up Your Workspace (5 min)

### Step 1: Create your project folder

Create a new folder somewhere you'll remember (e.g. your Desktop) and name it `novagrid-demo`.

Then open VS Code and go to **File → Open Folder...**, and select the `novagrid-demo` folder you just created. (Need a refresher? See [Part 5 of the Installation Guide](installation-guide.md).)

### Step 2: Create the starter file

In VS Code, go to **File → New File...**, name it `index.html`, and save it inside `novagrid-demo`.

Paste the following into it. This is what the freelancer left behind. Yes, it's bad. That's the point.

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NovaGrid</title>
    <style>
        /* TODO: add styles I guess? */
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 0;
        }
    </style>
</head>
<body>

    <header>
        <h1>NovaGrid</h1>
        <p>we do energy stuff</p>
        <!-- TODO: navigation maybe? -->
    </header>

    <main>
        <section>
            <h2>Welcome</h2>
            <p>Lorem ipsum dolor sit amet. This is placeholder text that the developer
            definitely meant to replace before disappearing forever.</p>
            <a href="#">Click here for something</a>
        </section>

        <section>
            <h2>Our Impact</h2>
            <p>We have saved lots of energy. Like, a LOT. Trust us.</p>
            <!-- TODO: some kind of numbers here? investors like numbers -->
        </section>
    </main>

    <footer>
        <p>© NovaGrid 2025. All rights reserved probably.</p>
    </footer>

</body>
</html>
```

### See it in the browser

To check what the page actually looks like at any point, open `index.html` in your web browser:

- **Easiest way:** find `novagrid-demo` in your file explorer (Finder on Mac, File Explorer on Windows), then double-click `index.html`. It'll open in your default browser.
- **From VS Code:** right-click `index.html` in the Explorer sidebar and choose **Reveal in File Explorer** (Windows) / **Reveal in Finder** (Mac) — then double-click the file from there.

Whenever Claude makes a change later in this exercise, just refresh that browser tab to see the update — no need to reopen the file each time.

### Step 3: Initialize Git

Open the integrated terminal in VS Code (**View → Terminal**, or `` Ctrl+` ``), then run:

```bash
git init
git add .
git commit -m "chore: initial commit, inherited from missing developer"
```

Elena will see this commit. She'll understand.

---

## 📋 Phase 1: Write the Handover Doc — `CLAUDE.md` (5 min)

Before you wake up your AI engineer, you need to write a proper handover document. The freelancer didn't leave one (of course), so you'll write the one you wish they had.

Create a file called `CLAUDE.md` in your project root.

> **Why this matters:** Claude Code reads this file automatically when it starts. Think of it as the onboarding doc for your engineer. A good `CLAUDE.md` means fewer misunderstandings, less wasted context, and a calmer night for everyone.

### Your `CLAUDE.md` should include the following sections:

Copy the template below into your `CLAUDE.md` file:

```markdown
# NovaGrid: Investor Demo Page

## Your Role
You are a senior front-end engineer supporting a non-technical Head of Product
(that's me) overnight, before a critical investor demo at 9 AM.
You are calm, careful, and you've seen rushed launches go badly before.
You'd rather ship something solid than something flashy and broken.

## The Project
- Single static page: `index.html`
- Inline CSS and JS only (no external files, no frameworks, no npm, no build tools)
- The page must work by simply opening the file in a browser (double-click to open)
- Audience: investors seeing NovaGrid for the first time
- Tone: clean, confident, professional but not boring

## How You Work (Golden Rules)

### Rule 1: Orient Before You Act
Before making ANY change, briefly summarize:
  - What the relevant file currently looks like
  - What you understand the request to be
Confirm with me before writing code.

### Rule 2: One Change at a Time
Implement exactly what was asked. Nothing extra.
No "while I'm at it" additions. No unrequested libraries.
No surprise refactors. One ticket, one change.

### Rule 3: Explain in Plain Language
Before showing code, tell me in simple terms:
  - What you're about to change
  - Why this approach makes sense
I'm not a developer. Skip the jargon. If you must use a technical term,
explain it in the same sentence.

### Rule 4: Ask, Don't Guess
If a request is vague or ambiguous, ask ONE clarifying question
instead of making assumptions. Better to ask than to rebuild.

### Rule 5: Fail Gracefully
If something breaks:
  1. STOP immediately
  2. Tell me clearly what broke and why (in plain language)
  3. Recommend reverting to the last working state via git
  4. Do NOT attempt to patch on top of broken code
  5. Wait for my go-ahead before trying again
If a second attempt also fails, revert again and suggest
a simpler alternative approach.

### Rule 6: Hands Off
Never modify these files unless I explicitly ask:
  - `.git/` directory
  - `CLAUDE.md`
  - `.claude/` directory

### Rule 7: Commit Messages
Use short, imperative style with a prefix:
  - `feat: add hero section with CTA`
  - `fix: correct broken navigation link`
  - `chore: clean up unused CSS`

## Git Workflow
You are responsible for all git operations. When I give you a task:
  1. Create a feature branch (e.g. `feature/hero-section`) and switch to it
  2. Make the changes on that branch
  3. Wait for my approval before committing
  4. When I confirm, stage the changes, commit with a proper message,
     switch back to `main`, and merge the feature branch
  5. Never commit directly to `main`

If I ask you to revert, use `git checkout -- .` to discard changes on the
current branch. If already committed, use `git reset` to undo the commit
before switching branches.

## Communication Style
- Keep responses concise (I'm reading this at 2 AM)
- Use bullet points for lists, short paragraphs for explanations
- When reporting what you changed, give me a quick before/after summary
```

### Step 4: Commit your handover doc

```bash
git add CLAUDE.md
git commit -m "chore: add CLAUDE.md engineering handover"
```

### Step 5: Start Claude Code

```bash
claude
```

Welcome your engineer! As a warm-up, type something like:

> "Hey! Read through the project and tell me what we're working with. Don't change anything yet."

Claude should orient itself using `CLAUDE.md` and the existing `index.html`. Read its summary carefully. This is your engineer confirming it understood the handover doc.

> 💡 **Tip:** Run `/model` and check which model you're currently using. For the first couple of simple tasks, consider switching to a more affordable one. You can always switch back for complex stuff later.

---

## 🎫 Phase 2: The Tickets (15 min)

Tickets will arrive from different NovaGrid stakeholders. For **each ticket**, you will ask Claude Code to handle the full workflow: branching, implementing, committing, and merging. You're the Head of Product, not the engineer. Let Claude do the git work too.

Here's how a ticket cycle looks inside Claude Code:

**Step 1: Tell Claude to create a branch and start working.**
Prompt Claude something like:

> *"Create a new branch called feature/hero-section, then [paste the ticket here]"*

Claude will create the branch, switch to it, and implement the change.

**Step 2: Review the result.**
Open `index.html` in your browser and check the result visually. Does it look right? Does it match what the stakeholder asked for?

**Step 3: If it looks good, tell Claude to commit and merge.**
Prompt Claude:

> *"Looks good. Commit this with an appropriate message, then switch back to main and merge the branch."*

Claude will handle `git add`, `git commit`, `git checkout main`, and `git merge` for you.

**Step 4: If something looks wrong, tell Claude to revert.**
Prompt Claude:

> *"This doesn't look right. Revert the changes and let's try a different approach."*

Remember Rule 5 from `CLAUDE.md`: don't pile fixes on top of broken code. Revert first, then re-prompt.

**Step 5: Check your context.**
Run `/context` after each ticket.

> 🧠 **Context Rule of Thumb:**
> After each ticket, run `/context`. If your context is getting heavy and the next ticket is unrelated to what you just did, consider using `/clear` (Claude will re-read `CLAUDE.md` fresh). If the next ticket builds on the current one, try `/compact` instead to keep continuity while freeing space.

---

### 🎫 Ticket #1: The Hero Section
**From:** Mara (CEO)
**Priority:** High
**Message:**

> *"The hero section is embarrassing. We need something that actually tells people what NovaGrid does. We're a clean-energy analytics platform that helps businesses track, reduce, and report their carbon footprint. Make it inspiring. Add a call-to-action button that invites people to request a demo. I trust your taste, just make it look like we know what we're doing."*

**Your move:** This is open-ended on purpose. You decide how to translate Mara's vision into a prompt for Claude. Think about what details would help Claude deliver something good on the first try.

> 📌 **Example prompt to get started:**
> *"Create a branch called feature/hero-section. Then redesign the hero section of index.html so that [your creative direction here based on Mara's message]."*
>
> Once you're happy with the result:
> *"Commit this and merge it back into main."*

After merging, check `/context`.

---

### 🎫 Ticket #2: Brand Identity
**From:** Théo (Head of Design)
**Priority:** High
**Message:**

> *"Here are our brand guidelines. Please match these."*

Théo didn't write a spec. He sent a screenshot (typical designer move).

**Your move:**
1. Find an image that could serve as NovaGrid's brand reference (a color palette, a logo, any visual). You can use the provided placeholder below or grab something from the web that feels "clean energy startup."
2. **Paste or drag the image directly into the Claude Code terminal.**
3. Prompt Claude something like:

> *"Create a branch called feature/brand-identity. Here's our brand reference [paste/drag your image]. Update the styling of index.html to match these colors and visual direction."*

Once you're happy: *"Commit and merge back to main."*

> 🖼️ **This is your chance to practice giving Claude visual context!** Drag and drop an image right into the terminal, or use your system's paste shortcut.

After merging, run `/context` again. If it's getting heavy, this is a natural point for `/clear`, since the next ticket is completely unrelated to styling.

---

### 🎫 Ticket #3: Legal Compliance
**From:** Priya (Legal)
**Priority:** Medium
**Message:**

> *"We need a proper footer. Include a copyright notice for the current year, a link to a Privacy Policy page (just use # as the href for now), and a short cookie disclaimer. Standard language is fine. Also, GDPR requires a 'Manage Preferences' link. Keep it professional."*

**Your move:** This is a straightforward text task. Low complexity, low risk.

> 💰 **Budget moment!** This is a great time to switch to a cheaper model. Run `/model` and pick something lightweight. Why burn premium tokens on a footer?

> 📌 **Prompt Claude:**
> *"Create a branch called feature/legal-footer. Then [paste Priya's request]. Commit and merge when done."*
>
> Since this is simple and well-defined, you can ask Claude to do everything in one go.

After merging, switch your model back to the more capable one. The next ticket needs it.

---

### 🎫 Ticket #4: The "Wow Factor"
**From:** Sam (Investor Liaison)
**Priority:** HIGH (all caps, his choice)
**Message:**

> *"The investors LOVE numbers. Can we add a section that shows NovaGrid's impact in a visually impressive way? Think big counters or stats, something that catches the eye when they scroll down. I'm thinking stuff like tons of CO2 offset, renewable energy monitored, companies onboarded... make it feel alive. Use your judgment on the design."*

**Your move:** This is deliberately vague. Sam wants "wow" but hasn't defined what that means. This is your chance to either:
- Ask Claude a clarifying question (as `CLAUDE.md` Rule 4 suggests), or
- Provide your own creative direction in the prompt

Think about it: what would actually impress an investor scrolling through this page?

> 📌 **Prompt Claude:**
> *"Create a branch called feature/impact-stats. Then [your interpretation of Sam's request, with your own creative direction]."*

> ⚠️ **Heads up:** if Claude tries to do something overly ambitious (importing external libraries, building a complex animation framework), remember Rule 2: one change at a time, no unrequested extras. If the page breaks, **don't panic and don't pile more prompts on top.** Tell Claude:
>
> *"This broke the page. Revert all changes on this branch, then let's try a simpler approach: [your new direction]."*

After merging, run `/context`. You've been through four tickets. This is likely a good moment for `/compact` or `/clear` before the next phase.

---

## 🔧 Phase 3: Build Your Own Command (5 min)

Elena (CTO) doesn't just want features. She wants **quality control**. Before she reviews anything, she expects you to have run a QA check.

Let's build a reusable command for that.

### Step 1: Create the commands directory

```bash
mkdir -p .claude/commands
```

### Step 2: Create your QA check command

Create a file at `.claude/commands/qa-check.md` with instructions for Claude to review the current state of `index.html`.

**What should this command do?** That's up to you! But here are some ideas for what it could check:
- Placeholder text still lurking (lorem ipsum, TODO comments, "click here for something")
- Missing alt text on images
- Broken or placeholder links (`href="#"`)
- Unused CSS rules
- Accessibility basics (color contrast, semantic HTML)
- Anything that would make Elena raise an eyebrow

> 🎨 **Be creative!** Write the instructions in a way that reflects how YOU would want a QA report delivered. Bullet points? Severity levels? A pass/fail summary? It's your command.

### Step 3: Commit and test it

```bash
git add .claude/
git commit -m "chore: add qa-check custom command"
```

Now run it in Claude:

```
/project:qa-check
```

Review the output. If Claude found issues, tell it:

> *"Create a branch called feature/qa-fixes and fix the issues you just found. Then commit and merge back to main."*

---

## 🏁 Phase 4: The Sunrise Review (5 min)

Elena is awake. Coffee in hand. Time for the final review.

### Ask Claude to show you the full picture:

> *"Show me the git log as a graph so I can see the full branch and merge history."*

Look at the output. Does it tell a clean story? Can Elena see what happened tonight, branch by branch, without needing to ask you a single question?

### Tag your release:

> *"Tag the current state as v1.0-investor-demo."*

### Open `index.html` in your browser one last time.

You shipped it. The investors will see this in a few hours. Take a moment to appreciate what you built in under an hour, with an AI engineer, a half-finished starter file, and zero sleep.

---

## ⭐ Bonus Challenges (if you have time)

### Bonus 1: Write Another Custom Command
Create `.claude/commands/ticket.md`, a command that, when run, asks Claude to generate a short changelog entry summarizing the last feature added. Test it after your final merge.

### Bonus 2: Create a Skill
Create a `.claude/skills/` directory and write a skill file (e.g., `brand-voice.md`) that teaches Claude how NovaGrid communicates: tone of voice, words to avoid, how to talk about sustainability without sounding generic. Then reference it in your next prompt and see how it changes Claude's output.

### Bonus 3: The 2 AM Curveball
Mara sends one more message at 2 AM:

> *"Actually, can we add a dark mode toggle? The lead investor apparently hates bright screens."*

Ship it. On a branch. With a clean commit. You know the drill by now.

---

## 🏠 Take Claude Home: Ideas to Try After This Workshop

You've now seen how Claude Code works in a structured workflow. Here are ways to keep exploring:

**For your daily work:**
- Use Claude Code to draft and iterate on documents, emails, or reports from the terminal
- Set up a `CLAUDE.md` for a personal project and see how it changes the quality of responses
- Build custom commands for tasks you repeat often (weekly reports, code reviews, data formatting)

**Ship it for real (deploy to Vercel):**

Why keep your page as a local file? You can deploy it live in minutes:

1. Create a free account at [vercel.com](https://vercel.com) if you don't have one
2. Install the Vercel CLI: run `npm install -g vercel` in your terminal
3. In your project folder, simply run: `vercel`
4. Follow the prompts (accept defaults). Done. Your page is live on the internet.

Or, even better, ask Claude Code to do it for you:

> *"Help me deploy this project to Vercel. Walk me through the setup."*

Now you can share the URL with your friends, your team, or your imaginary investors. 🚀

**For fun:**
- Build a personal portfolio page using the same branch-and-merge workflow
- Challenge yourself to build a small tool (a calculator, a to-do list, a quiz) with Claude as your engineer
- Try giving Claude a screenshot of a website you admire and ask it to recreate the layout

**For deeper learning:**
- Experiment with different models for different tasks and compare the results
- Practice writing increasingly detailed `CLAUDE.md` files and notice how it affects output quality
- Explore MCP (Model Context Protocol) integrations to connect Claude to your everyday tools

---

## 📊 Reflection Questions (for group discussion)

1. At what point did your `CLAUDE.md` save you from having to repeat instructions?
2. When did you switch models, and did it actually matter for the output quality?
3. How did you decide between `/clear` and `/compact`? What happened to Claude's behavior after each?
4. What would you add to your `CLAUDE.md` if you had to do this exercise again?
5. How does the branch-and-merge workflow change the way you think about giving instructions to an AI?

---

*Good luck tonight. Elena is counting on you. The investors are counting on NovaGrid. And somewhere out there, a freelance developer is sipping cocktails on a beach, blissfully unaware of the chaos they left behind.*

*You've got this.* 🚀
