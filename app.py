import os, re, math
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics.pairwise import cosine_similarity
try:
    from mlxtend.frequent_patterns import apriori, association_rules
    import inspect as _inspect
    _AR_NEEDS_NUM_ITEMSETS = 'num_itemsets' in _inspect.signature(association_rules).parameters
except ImportError:
    apriori = None
    association_rules = None
    _AR_NEEDS_NUM_ITEMSETS = False

app = Flask(__name__, static_folder='.', static_url_path='')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
CORS(app)

SKILLS = [
    'python','java','javascript','typescript','c++','c#','golang','rust','php','ruby','swift','kotlin','scala','r',
    'react','vue','angular','svelte','html','css','sass','tailwind','bootstrap','next.js','nuxt',
    'node.js','express','django','flask','fastapi','spring','laravel','rails',
    'sql','mysql','postgresql','mongodb','redis','sqlite','firebase','elasticsearch','dynamodb','nosql',
    'docker','kubernetes','aws','azure','gcp','terraform','linux','ci/cd','jenkins','github actions','ansible','git','github',
    'rest api','graphql','microservices','system design','agile','scrum','jira',
    'machine learning','deep learning','tensorflow','pytorch','pandas','numpy','scikit-learn','data analysis','tableau','power bi','excel',
    'figma','sketch','adobe xd','photoshop','illustrator','user research','prototyping','wireframing','typography','design systems','ux design','ui design','responsive design','a/b testing','accessibility',
    'communication','teamwork','leadership','management','problem solving','critical thinking','collaboration',
]

SECTION_KEYWORDS = {
    'skills': [
        'skill', 'technical skill', 'core skill', 'key skill', 'technology',
        'tech stack', 'competency', 'competencies', 'proficiency', 'proficiencies',
        'tools', 'languages', 'frameworks', 'expertise', 'technical expertise',
    ],
    'experience': [
        'experience', 'work experience', 'professional experience', 'employment',
        'work history', 'internship', 'internships', 'training', 'industrial training',
        'apprenticeship', 'position', 'role', 'job', 'career', 'occupation',
    ],
    'education': [
        'education', 'academic', 'qualification', 'degree', 'bachelor', 'master',
        'phd', 'diploma', 'university', 'college', 'school', 'b.tech', 'b.e',
        'm.tech', 'mba', 'bsc', 'msc', 'b.sc', 'm.sc', 'graduation',
    ],
    'projects': [
        'project', 'academic project', 'personal project', 'key project',
        'portfolio', 'work sample', 'built', 'developed', 'created', 'designed',
        'implemented', 'github', 'application', 'app', 'system', 'website',
    ],
    'summary': [
        'summary', 'objective', 'career objective', 'profile', 'about me',
        'professional summary', 'my goal', 'as a ', 'i aim to', 'i am a',
        'motivated', 'seeking to', 'apply my skills', 'looking for opportunit',
        'overview', 'career goal', 'professional profile',
    ],
}

SKILL_META = {
    'docker':           {'emoji':'🐳','bg':'linear-gradient(135deg,#0db7ed,#384d54)'},
    'kubernetes':       {'emoji':'☸️','bg':'linear-gradient(135deg,#326ce5,#1a3a8f)'},
    'aws':              {'emoji':'☁️','bg':'linear-gradient(135deg,#ff9900,#c45000)'},
    'azure':            {'emoji':'☁️','bg':'linear-gradient(135deg,#0078d4,#004578)'},
    'python':           {'emoji':'🐍','bg':'linear-gradient(135deg,#3776ab,#ffd343)'},
    'machine learning': {'emoji':'🤖','bg':'linear-gradient(135deg,#8e44ad,#c0392b)'},
    'react':            {'emoji':'⚛️','bg':'linear-gradient(135deg,#61dafb,#21232a)'},
    'sql':              {'emoji':'🗄️','bg':'linear-gradient(135deg,#f7971e,#ffd200)'},
    'rest api':         {'emoji':'🔗','bg':'linear-gradient(135deg,#11998e,#38ef7d)'},
    'system design':    {'emoji':'🏗️','bg':'linear-gradient(135deg,#8e44ad,#c0392b)'},
    'ci/cd':            {'emoji':'⚙️','bg':'linear-gradient(135deg,#2c3e50,#4ca1af)'},
    'git':              {'emoji':'🔀','bg':'linear-gradient(135deg,#f05032,#333)'},
    'javascript':       {'emoji':'⚡','bg':'linear-gradient(135deg,#f7df1e,#c8b400)'},
    'typescript':       {'emoji':'📘','bg':'linear-gradient(135deg,#3178c6,#1e4f8c)'},
    'java':             {'emoji':'☕','bg':'linear-gradient(135deg,#f89820,#5382a1)'},
    'node.js':          {'emoji':'🟩','bg':'linear-gradient(135deg,#68a063,#215732)'},
    'mongodb':          {'emoji':'🍃','bg':'linear-gradient(135deg,#47a248,#1a5c1e)'},
    'postgresql':       {'emoji':'🐘','bg':'linear-gradient(135deg,#336791,#1a3a5c)'},
    'figma':            {'emoji':'🎨','bg':'linear-gradient(135deg,#f24e1e,#a259ff)'},
    'graphql':          {'emoji':'💡','bg':'linear-gradient(135deg,#e10098,#7b0055)'},
    'go':               {'emoji':'🐹','bg':'linear-gradient(135deg,#00add8,#007d9c)'},
    'rust':             {'emoji':'🦀','bg':'linear-gradient(135deg,#dea584,#b7410e)'},
    'ruby':             {'emoji':'💎','bg':'linear-gradient(135deg,#cc342d,#8a0e05)'},
    'php':              {'emoji':'🐘','bg':'linear-gradient(135deg,#777bb4,#4f5b93)'},
    'swift':            {'emoji':'🐦','bg':'linear-gradient(135deg,#f05138,#d1341c)'},
    'kotlin':           {'emoji':'🤖','bg':'linear-gradient(135deg,#7f52ff,#d91fae)'},
    'c++':              {'emoji':'⚙️','bg':'linear-gradient(135deg,#00599c,#004482)'},
    'angular':          {'emoji':'🅰️','bg':'linear-gradient(135deg,#dd0031,#c3002f)'},
    'vue':              {'emoji':'🟩','bg':'linear-gradient(135deg,#4fc08d,#35495e)'},
    'redis':            {'emoji':'🔴','bg':'linear-gradient(135deg,#d82c20,#9c1c14)'},
    'firebase':         {'emoji':'🔥','bg':'linear-gradient(135deg,#ffca28,#f57c00)'},
    'html':             {'emoji':'🌐','bg':'linear-gradient(135deg,#e34f26,#b53f1d)'},
    'css':              {'emoji':'🎨','bg':'linear-gradient(135deg,#1572b6,#1b5585)'},
    'tailwind':         {'emoji':'🌊','bg':'linear-gradient(135deg,#38b2ac,#2c7a7b)'},
    'pandas':           {'emoji':'🐼','bg':'linear-gradient(135deg,#150458,#090226)'},
    'numpy':            {'emoji':'🔢','bg':'linear-gradient(135deg,#4d77cf,#013220)'},
    'jenkins':          {'emoji':'👨‍🍳','bg':'linear-gradient(135deg,#d33833,#335061)'},
    'gcp':              {'emoji':'☁️','bg':'linear-gradient(135deg,#4285f4,#0f9d58)'}
}
DEFAULT_META = {'emoji':'🛠️','bg':'linear-gradient(135deg,#005f98,#2aa7ff)'}

# ── Helpers ───────────────────────────────────────────────────────────────────
def extract_identity(text):
    """Extract name / email / phone from raw resume text."""
    # — Email (first match) —
    email_match = re.search(r'[\w.+\-]+@[\w.\-]+\.[a-zA-Z]{2,}', text)
    email = email_match.group(0) if email_match else '-'

    # — Phone (10-digit, optionally with spaces/dashes/country code) —
    phone_match = re.search(
        r'(?:\+?\d[\s\-]?)?(\d[\s\-]?){9}\d',
        text
    )
    if phone_match:
        raw_phone = phone_match.group(0)
        # Keep only digits, then re-format
        digits = re.sub(r'\D', '', raw_phone)
        phone = digits[-10:] if len(digits) >= 10 else digits
    else:
        phone = '-'

    # — Name (first non-empty line that looks like a name) —
    name = 'Unknown'
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        # Skip lines that look like emails/phones/dates/urls
        if re.search(r'[@:\d\/\.\|]', line):
            continue
        words = line.split()
        # A name is usually 1-4 capitalised words with no common resume keywords
        resume_stop = {'resume', 'cv', 'curriculum', 'vitae', 'profile',
                       'objective', 'summary', 'linkedin', 'github'}
        if (1 <= len(words) <= 5
                and all(w[0].isupper() for w in words if w)
                and not any(w.lower() in resume_stop for w in words)):
            name = ' '.join(words)
            break

    return {'name': name, 'email': email, 'phone': phone}

def extract_text(file):
    filename = file.filename.lower()
    if filename.endswith('.pdf'):
        try:
            import pdfplumber
            with pdfplumber.open(file) as pdf:
                return '\n'.join(p.extract_text() or '' for p in pdf.pages)
        except Exception as e:
            app.logger.warning(f"PDF extraction failed for {file.filename}: {e}")
            return ''
    elif filename.endswith('.docx') or filename.endswith('.doc'):
        try:
            from docx import Document
            doc = Document(file)
            return '\n'.join(p.text for p in doc.paragraphs)
        except Exception as e:
            app.logger.warning(f"DOCX extraction failed for {file.filename}: {e}")
            return ''
    return ''

def detect_skills(text):
    lower = text.lower()
    found = []
    for skill in SKILLS:
        pattern = r'\b' + re.escape(skill) + r'\b'
        if re.search(pattern, lower):
            found.append(skill)
    return list(dict.fromkeys(found))

def detect_sections(text):
    lower = text.lower()
    found = {}
    for section, keywords in SECTION_KEYWORDS.items():
        found[section] = any(kw in lower for kw in keywords)
    if not found['summary']:
        top_words = text.split()[:150]
        top_text  = ' '.join(top_words)
        sentences = re.split(r'[.!?\n]', top_text)
        if any(len(s.split()) >= 20 for s in sentences):
            found['summary'] = True
    return found

def validate_resume(text):
    lower = text.lower()
    required = ['skill','education','experience','project']
    hits = sum(1 for kw in required if kw in lower)
    return hits >= 2

def build_recommendations(missing_skills, limit=8):
    recs = []
    for skill in missing_skills[:limit]:
        meta = SKILL_META.get(skill, DEFAULT_META)
        recs.append({
            'skill': skill,
            'emoji': meta['emoji'],
            'bg': meta['bg'],
            'youtube_url': f"https://www.youtube.com/results?search_query={skill.replace(' ','+').replace('/','')}+tutorial+for+beginners",
            'project_url': f"https://www.google.com/search?q={skill.replace(' ','+').replace('/','')}+project+ideas",
        })
    return recs

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/<path:filename>')
def static_files(filename):
    if filename.startswith('api/'):
        return jsonify({'error': 'Not found'}), 404
    return send_from_directory('.', filename)

@app.route('/api/analyze', methods=['POST'])
def analyze():
    if request.method != 'POST':
        return jsonify({'error': 'Method not allowed. Use POST.'}), 405
    if 'resume' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['resume']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    text = extract_text(file)
    if not text.strip():
        return jsonify({'error': 'Could not extract text from file. Make sure the PDF has selectable text.'}), 422

    if not validate_resume(text):
        return jsonify({'error': 'Please upload a valid resume. The document should contain sections like Skills, Education, Experience, and Projects.'}), 422

    sections = detect_sections(text)
    missing_sections = [s for s, present in sections.items() if not present]

    detected_skills = detect_skills(text)
    all_important = ['python','javascript','typescript','react','node.js','sql','docker','git','aws','machine learning','rest api','system design','ci/cd','kubernetes','mongodb','postgresql','java','figma','graphql']
    missing_skills = [s for s in all_important if s not in detected_skills]

    skill_score = min(100, len(detected_skills) * 5)
    proj_score  = 85 if sections.get('projects') else 20
    exp_score   = 85 if sections.get('experience') else 15
    edu_score   = 90 if sections.get('education') else 30
    word_count  = len(text.split())
    fmt_score   = min(100, max(20, (word_count // 10)))

    resume_score = round(
        skill_score * 0.30 +
        proj_score  * 0.25 +
        exp_score   * 0.25 +
        edu_score   * 0.10 +
        fmt_score   * 0.10
    )
    resume_score = max(10, min(95, resume_score))

    section_scores = {
        'Skills & Technologies': skill_score,
        'Work Experience':       exp_score,
        'Education':             edu_score,
        'Projects':              proj_score,
        'Summary / Objective':   80 if sections.get('summary') else 20,
    }

    recommendations = build_recommendations(missing_skills)

    return jsonify({
        'resume_score':    resume_score,
        'detected_skills': detected_skills,
        'missing_skills':  missing_skills,
        'missing_sections': missing_sections,
        'section_scores':  section_scores,
        'recommendations': recommendations,
        'filename':        file.filename,
    })


@app.route('/api/match', methods=['POST'])
def match():
    print("[DEBUG] start request")
    # ── Outer try/except — always returns JSON, never hangs ──────────────────
    try:
        if request.method != 'POST':
            return jsonify({'error': 'Method not allowed. Use POST.'}), 405

        files = request.files.getlist('files')
        if not files:
            return jsonify({'error': 'No files uploaded'}), 400

        if 'job_description' not in request.form:
            return jsonify({'error': 'No job description provided'}), 400

        job_desc = request.form['job_description']
        if not job_desc.strip():
            return jsonify({'error': 'Empty job description'}), 400

        print(f"[DEBUG] Received {len(files)} file(s)")
        app.logger.info(f"[MATCH] Received {len(files)} file(s)")

        jd_skills = set(detect_skills(job_desc))
        if not jd_skills:
            jd_skills = set(detect_skills(job_desc.lower()))

        print(f"[DEBUG] JD skills detected: {jd_skills}")
        app.logger.info(f"[MATCH] JD skills: {jd_skills}")
        total_jd = len(jd_skills)

        # ── PHASE 1: BASE RESULTS (always runs first, independent of ML) ─────
        results = []
        valid_result_indices = []   # indices into results[] for ML enhancements
        candidate_skill_vectors = []
        candidate_features = []

        for file in files:
            if file.filename == '':
                continue

            text = extract_text(file)
            print(f"[DEBUG] extraction done for: {file.filename} — chars={len(text)}")

            if not text.strip():
                # Extraction failed — still produce a safe fallback result
                app.logger.warning(f"[MATCH] Extraction failed: {file.filename}")
                results.append({
                    'filename':           file.filename,
                    'name':               'Unknown',
                    'email':              '-',
                    'phone':              '-',
                    'match_score':        15,
                    'skill_score':        0,
                    'project_score':      20,
                    'exp_score':          15,
                    'candidate_exp':      0,
                    'has_edu':            False,
                    'matched_skills':     [],
                    'missing_skills':     sorted(jd_skills),
                    'recommendations':    build_recommendations(sorted(jd_skills)),
                    'total_jd_skills':    total_jd,
                    'skills_list':        [],
                    'cluster':            'General Profile',
                    'classification':     'Average',
                    'similar_candidates': [],
                    'extraction_failed':  True,
                })
                continue  # no ML vectors for failed files

            # ── Scoring ──────────────────────────────────────────────────────
            try:
                identity       = extract_identity(text)
                sections       = detect_sections(text)
                detected_skills = detect_skills(text)
                resume_skills  = set(detected_skills)
                matched_skills = sorted(resume_skills & jd_skills)
                missing_skills = sorted(jd_skills - resume_skills)

                match_score = round(max(15, min(95,
                    (len(matched_skills) / total_jd * 100) if total_jd > 0 else 50
                )))

                skill_score   = min(100, len(detected_skills) * 5)
                proj_score    = 85 if sections.get('projects') else 20
                exp_score     = 85 if sections.get('experience') else 15
                has_edu       = sections.get('education', False)

                exp_matches   = re.findall(r'(\d+)\s*(?:year|yr)', text.lower())
                candidate_exp = max([int(m) for m in exp_matches] + [0]) if exp_matches else 1

                # Rule-based classification (fast, replaces sklearn Decision Tree)
                if match_score > 75:
                    classification = 'Excellent'
                elif match_score > 50:
                    classification = 'Good'
                else:
                    classification = 'Average'

                print(f"[DEBUG] scored {file.filename}: match={match_score}, class={classification}")

                vector = [1 if skill in resume_skills else 0 for skill in SKILLS]
                valid_result_indices.append(len(results))
                candidate_skill_vectors.append(vector)
                candidate_features.append([match_score, len(matched_skills)])

                results.append({
                    'filename':           file.filename,
                    'name':               identity['name'],
                    'email':              identity['email'],
                    'phone':              identity['phone'],
                    'match_score':        match_score,
                    'skill_score':        skill_score,
                    'project_score':      proj_score,
                    'exp_score':          exp_score,
                    'candidate_exp':      candidate_exp,
                    'has_edu':            has_edu,
                    'matched_skills':     matched_skills,
                    'missing_skills':     missing_skills,
                    'recommendations':    build_recommendations(missing_skills),
                    'total_jd_skills':    total_jd,
                    'skills_list':        list(resume_skills),
                    'cluster':            'General Profile',  # ML may override
                    'classification':     classification,     # ML may override
                    'similar_candidates': [],
                })

            except Exception as score_err:
                app.logger.warning(f"[MATCH] Scoring failed for {file.filename}: {score_err}")
                results.append({
                    'filename':           file.filename,
                    'name':               'Unknown',
                    'email':              '-',
                    'phone':              '-',
                    'match_score':        15,
                    'skill_score':        0,
                    'project_score':      20,
                    'exp_score':          15,
                    'candidate_exp':      0,
                    'has_edu':            False,
                    'matched_skills':     [],
                    'missing_skills':     sorted(jd_skills),
                    'recommendations':    build_recommendations(sorted(jd_skills)),
                    'total_jd_skills':    total_jd,
                    'skills_list':        [],
                    'cluster':            'General Profile',
                    'classification':     'Average',
                    'similar_candidates': [],
                })

        print(f"[DEBUG] base results built: {len(results)} candidate(s), {len(valid_result_indices)} valid for ML")
        app.logger.info(f"[MATCH] Valid candidates: {len(valid_result_indices)} / Total: {len(results)}")

        # ── PHASE 2: ML ENHANCEMENTS (purely optional — never block core) ────
        final_rules = []
        n_valid = len(valid_result_indices)

        # ── DSBD 1: KMeans clustering ─────────────────────────────────────────
        print("[DEBUG] before KMeans")
        if n_valid >= 2:
            try:
                vec_array = np.array(candidate_skill_vectors, dtype=float)
                if np.all(vec_array.sum(axis=1) == 0):
                    raise ValueError("All vectors are zero")
                n_clusters = min(4, n_valid)
                labels = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto').fit_predict(vec_array)
                cmap = {0: "Frontend Focused", 1: "Backend Focused", 2: "Balanced Profile", 3: "Entry Level"}
                for vi, ri in enumerate(valid_result_indices):
                    results[ri]['cluster'] = cmap.get(int(labels[vi]), "General Profile")
                app.logger.info("[KMEANS] Done ✓")
            except Exception as e:
                app.logger.warning(f"[KMEANS] Fallback: {e}")

        # ── DSBD 2: Classification already applied via rules in Phase 1 ───────

        # ── DSBD 3: Apriori association rules ─────────────────────────────────
        print("[DEBUG] before Apriori")
        if apriori and n_valid >= 2:
            try:
                from collections import Counter
                all_skills   = [results[ri]['skills_list'] for ri in valid_result_indices]
                skill_counts = Counter(s for lst in all_skills for s in lst)
                unique_sk    = sorted(s for s, _ in skill_counts.most_common(25))
                if len(unique_sk) >= 2:
                    skill_df = pd.DataFrame(
                        [{s: (s in c) for s in unique_sk} for c in all_skills],
                        dtype=bool
                    )
                    freq = apriori(skill_df, min_support=0.3, use_colnames=True)
                    if not freq.empty:
                        ar_kwargs = dict(metric="confidence", min_threshold=0.7)
                        if _AR_NEEDS_NUM_ITEMSETS:
                            ar_kwargs['num_itemsets'] = len(skill_df)
                        rules = association_rules(freq, **ar_kwargs)
                        for _, row in rules.iterrows():
                            ant  = list(row['antecedents'])[0]
                            con  = list(row['consequents'])[0]
                            conf = round(row['confidence'] * 100)
                            final_rules.append(f"{ant} → {con} ({conf}%)")
                app.logger.info(f"[APRIORI] Done ✓ — {len(final_rules)} rule(s)")
            except Exception as e:
                app.logger.warning(f"[APRIORI] Skipped: {e}")

        # ── DSBD 4: Cosine similarity ─────────────────────────────────────────
        print("[DEBUG] before Cosine Similarity")
        if n_valid >= 2:
            try:
                vec_array  = np.array(candidate_skill_vectors, dtype=float)
                sim_matrix = cosine_similarity(vec_array)
                for vi, ri in enumerate(valid_result_indices):
                    sorted_idx = sim_matrix[vi].argsort()[::-1]
                    similar = []
                    for si in sorted_idx:
                        if si == vi:
                            continue
                        if sim_matrix[vi][si] > 0.5:
                            similar.append(results[valid_result_indices[si]]['filename'])
                        if len(similar) >= 2:
                            break
                    results[ri]['similar_candidates'] = similar
                app.logger.info("[COSINE SIM] Done ✓")
            except Exception as e:
                app.logger.warning(f"[COSINE SIM] Skipped: {e}")

        # ── PHASE 3: HARD FINAL FALLBACK — results must NEVER be empty ───────
        if not results:
            print("[DEBUG] FINAL FALLBACK triggered — results was empty!")
            for file in files:
                if file.filename:
                    results.append({
                        'filename':           file.filename,
                        'name':               'Unknown',
                        'email':              '-',
                        'phone':              '-',
                        'match_score':        15,
                        'skill_score':        0,
                        'project_score':      20,
                        'exp_score':          15,
                        'candidate_exp':      0,
                        'has_edu':            False,
                        'matched_skills':     [],
                        'missing_skills':     sorted(jd_skills),
                        'recommendations':    build_recommendations(sorted(jd_skills)),
                        'total_jd_skills':    total_jd,
                        'skills_list':        [],
                        'cluster':            'General Profile',
                        'classification':     'Average',
                        'similar_candidates': [],
                    })

        print(f"[DEBUG] before return — {len(results)} result(s)")
        app.logger.info(f"[MATCH] Sending response — {len(results)} result(s)")
        return jsonify({
            'results':           results,
            'job_description':   job_desc,
            'association_rules': final_rules[:5],
        })

    except Exception as e:
        print("ERROR:", str(e))
        app.logger.exception(f"[MATCH] Unhandled exception: {e}")
        return jsonify({
            'results':           [],
            'association_rules': [],
            'error':             str(e)
        }), 200


# ── Global error handlers ─────────────────────────────────────────────────────
@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({'error': 'Method not allowed. This endpoint requires POST.'}), 405

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large. Please upload a file under 16 MB.'}), 413

@app.errorhandler(500)
def server_error(e):
    return jsonify({'error': f'Internal server error: {str(e)}'}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)
