# Reflection on Automated Code Generation for Multi-Agent Applications

I used Cursor, heavily leveraging the code gen capabilities for this work. Overall it provided a significant productivity boost; but there are a few things I would do differently the next time!

## 🚀 Impact
- Overall development speed increased by **2–5x** using automated code generation tools.

---

## ✅ What Went Well
- **Framework Code Generation**  
  Core structure and boilerplate were generated quickly and accurately. A lot of the code worked fine out of the box.

- **Supporting Code**  
  Generated high-quality auxiliary code with minimal manual work:
  - ✅ Unit and integration **tests**
  - ✅ Developer **scripts**
  - ✅ **Documentation**
  - ✅ **CI/CD pipelines**
  - ✅ **Pre-commit hooks**

---

## ⚠️ What Could Have Gone Better
- **Slower Code Acceptance**  
  - Delays in reviewing and accepting generated code introduced bottlenecks.

- **Regressions**
  - I ran into a lot of problems with **citation generation**:
    - The initial implementation was a rule based approach had a lot flaws. Eventually I switched to a LLM based approach for generating / inserting citations (duh!).
    - Multiple fixes via *cursor* were attempted but ineffective. Code generation made highly localized changes addressing the problem. Many of these lead to regressions and often did not fix the issue.
    - Ultimately had to change the approach to citation handling quite drastically (rule -> llm, section wise processing -> citations generation AFTER text generation). 
  - Writing tests earlier could have caught regressions sooner.

---

## 🧠 Lessons Learned

1. **Design First**
   - Sketch out **agent roles** and responsibilities.
   - Define **inter-agent communication protocols (signatures)**.

2. **Code Generation Process**
   - Use code generation to scaffold components.
   - **Always review** generated code before accepting or deploying. This is somewhat easier said than done because you might end up with a lot of generated and partially working code initially and might be tempted to get some working pieces first.

3. **Testing & Evaluation**
   - Write **tests and evaluation logic early**.
   - Use them to validate functionality and detect regressions.

4. **Iterative Development**
   - Treat generated code as a **starting point**.
   - Improve iteratively with testing and feedback.

---

