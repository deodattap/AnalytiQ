import os, re, math
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

app = Flask(__name__, static_folder='.', static_url_path='')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload
CORS(app)

# ── Predefined skill list (50 skills) ────────────────────────────────────────
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
    'skills':      ['skill','technologies','tech stack','tools','proficiencies','competencies','expertise'],
    'experience':  ['experience','employment','work history','professional','internship','intern','position','role','job'],
    'education':   ['education','degree','university','college','bachelor','master','phd','diploma','academic','school'],
    'projects':    ['project','built','developed','created','portfolio','github','application','app','system','website'],
    'summary':     ['summary','objective','profile','about','overview','career goal'],
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
}
DEFAULT_META = {'emoji':'🛠️','bg':'linear-gradient(135deg,#005f98,#2aa7ff)'}

# ── Helpers ───────────────────────────────────────────────────────────────────
def extract_text(file):
    filename = file.filename.lower()
    if filename.endswith('.pdf'):
        try:
            import pdfplumber
            with pdfplumber.open(file) as pdf:
                return '\n'.join(p.extract_text() or '' for p in pdf.pages)
        except Exception as e:
            return ''
    elif filename.endswith('.docx') or filename.endswith('.doc'):
        try:
            from docx import Document
            doc = Document(file)
            return '\n'.join(p.text for p in doc.paragraphs)
        except Exception as e:
            return ''
    return ''

def detect_skills(text):
    lower = text.lower()
    found = []
    for skill in SKILLS:
        pattern = r'\b' + re.escape(skill) + r'\b'
        if re.search(pattern, lower):
            found.append(skill)
    return list(dict.fromkeys(found))  # deduplicate preserving order

def detect_sections(text):
    lower = text.lower()
    found = {}
    for section, keywords in SECTION_KEYWORDS.items():
        found[section] = any(kw in lower for kw in keywords)
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
    # Never intercept API paths — return 404 so the API routes handle them properly
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

    # Detect sections
    sections = detect_sections(text)
    missing_sections = [s for s, present in sections.items() if not present]

    # Detect skills
    detected_skills = detect_skills(text)
    all_important = ['python','javascript','typescript','react','node.js','sql','docker','git','aws','machine learning','rest api','system design','ci/cd','kubernetes','mongodb','postgresql','java','figma','graphql']
    missing_skills = [s for s in all_important if s not in detected_skills]

    # Weighted score
    skill_score = min(100, len(detected_skills) * 5)
    proj_score  = 85 if sections.get('projects') else 20
    exp_score   = 85 if sections.get('experience') else 15
    edu_score   = 90 if sections.get('education') else 30
    # Formatting heuristic: length and variety
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
    if request.method != 'POST':
        return jsonify({'error': 'Method not allowed. Use POST.'}), 405
    if 'resume' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    if 'job_description' not in request.form:
        return jsonify({'error': 'No job description provided'}), 400

    file = request.files['resume']
    job_desc = request.form['job_description']

    text = extract_text(file)
    if not text.strip():
        return jsonify({'error': 'Could not extract text from the resume file.'}), 422

    resume_skills = set(detect_skills(text))
    jd_skills     = set(detect_skills(job_desc))

    if not jd_skills:
        # Fallback: extract any recognizable nouns from JD
        jd_skills = set(detect_skills(job_desc.lower()))

    matched_skills = sorted(resume_skills & jd_skills)
    missing_skills = sorted(jd_skills - resume_skills)
    total_jd = len(jd_skills)

    if total_jd > 0:
        raw_score = (len(matched_skills) / total_jd) * 100
    else:
        raw_score = 50

    match_score = round(max(15, min(95, raw_score)))

    recommendations = build_recommendations(missing_skills)

    return jsonify({
        'match_score':     match_score,
        'matched_skills':  matched_skills,
        'missing_skills':  missing_skills,
        'recommendations': recommendations,
        'filename':        file.filename,
        'total_jd_skills': total_jd,
    })


# ── Global error handlers (always return JSON, never HTML) ────────────────────
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
