

# ═══════════════════════════════════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/<path:filename>')
def static_files(filename):
    if filename.startswith('api/'):
        return jsonify({'error': 'Not found'}), 404
    return send_from_directory('.', filename)


# ── /api/analyze ─────────────────────────────────────────────────────────────
@app.route('/api/analyze', methods=['POST'])
def analyze():
    if 'resume' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['resume']
    if not file.filename:
        return jsonify({'error': 'Empty filename'}), 400

    text = extract_text(file)
    if not text.strip():
        return jsonify({'error': 'Could not extract text. Ensure the PDF has selectable text.'}), 422
    if not validate_resume(text):
        return jsonify({'error': 'Upload a valid resume with Skills, Experience, Education, Projects.'}), 422

    identity      = extract_identity(text)
    sections      = detect_sections(text)
    skill_conf    = detect_skills_sectioned(text)
    detected      = list(skill_conf.keys())
    exp_years     = extract_experience_years(text)
    edu_level     = detect_education_level(text)
    certs         = detect_certifications(text)

    important = [
        'python','javascript','typescript','react','node.js','sql','docker',
        'git','aws','machine learning','rest api','system design','ci/cd',
        'kubernetes','mongodb','postgresql','java','figma','graphql',
    ]
    missing = [s for s in important if s not in skill_conf]

    scores      = compute_match_score(detected, len(important), sections,
                                      exp_years, edu_level, certs, skill_conf,
                                      all_resume_skills=detected)
    skill_score = min(100, round(len(detected) / len(important) * 100))
    proj_score  = (95 if sections.get('projects') and certs
                   else 80 if sections.get('projects') else 20)
    exp_score   = (min(98, int(exp_years / 12 * 100))
                   if exp_years >= 1 else (45 if sections.get('experience') else 12))
    edu_score   = edu_level * 18

    edu_labels = {5:'PhD',4:'Masters',3:'Bachelors',2:'Diploma',1:'High School',0:'Unknown'}

    return jsonify({
        'resume_score':    scores['match_score'],
        'detected_skills': detected,
        'skill_confidence':skill_conf,
        'missing_skills':  missing,
        'missing_sections':[s for s,ok in sections.items() if not ok],
        'section_scores': {
            'Skills & Technologies': skill_score,
            'Work Experience':       exp_score,
            'Education':             edu_score,
            'Projects':              proj_score,
            'Summary / Objective':   80 if sections.get('summary') else 20,
        },
        'recommendations':  build_recommendations(missing),
        'filename':         file.filename,
        'name':             identity['name'],
        'email':            identity['email'],
        'phone':            identity['phone'],
        'linkedin':         identity['linkedin'],
        'github':           identity['github'],
        'experience_years': exp_years,
        'edu_level':        edu_level,
        'edu_label':        edu_labels.get(edu_level, 'Unknown'),
        'certifications':   certs,
    })


# ── /api/match ───────────────────────────────────────────────────────────────
@app.route('/api/match', methods=['POST'])
def match():
    print('[DEBUG] /api/match received')
    try:
        files = request.files.getlist('files')
        if not files:
            return jsonify({'error': 'No files uploaded'}), 400
        job_desc = request.form.get('job_description', '').strip()
        if not job_desc:
            return jsonify({'error': 'No job description provided'}), 400

        jd_skills = set(detect_skills(job_desc))
        total_jd  = len(jd_skills)
        print(f'[DEBUG] JD skills ({total_jd}): {sorted(jd_skills)}')

        results       = []
        valid_indices = []
        skill_vectors = []
        edu_labels    = {5:'PhD',4:'Masters',3:'Bachelors',2:'Diploma',
                         1:'High School',0:'Unknown'}

        for file in files:
            if not file.filename:
                continue
            text = extract_text(file)
            print(f'[DEBUG] {file.filename}: {len(text)} chars')
            if not text.strip():
                results.append(_fallback(file.filename, jd_skills, total_jd))
                continue
            try:
                identity   = extract_identity(text)
                sections   = detect_sections(text)
                skill_conf = detect_skills_sectioned(text)
                resume_set = set(skill_conf.keys())
                matched    = sorted(resume_set & jd_skills)
                missing    = sorted(jd_skills - resume_set)
                exp_years  = extract_experience_years(text)
                edu_level  = detect_education_level(text)
                certs      = detect_certifications(text)

                sd          = compute_match_score(matched, total_jd, sections,
                                                  exp_years, edu_level, certs, skill_conf,
                                                  all_resume_skills=sorted(resume_set))
                match_score = sd['match_score']
                skill_score = min(100, round(len(matched) / max(total_jd, 1) * 100)) if total_jd else min(100, len(resume_set) * 3)
                proj_score  = (95 if sections.get('projects') and len(certs) >= 2
                               else 80 if sections.get('projects')
                               else 35 if sections.get('experience') else 15)
                exp_score   = (min(98, int(exp_years / 12 * 100))
                               if exp_years >= 1 else (45 if sections.get('experience') else 12))
                has_edu     = bool(sections.get('education'))
                cls         = classify(match_score)
                cluster     = smart_cluster_label(resume_set)

                print(f'[DEBUG] {file.filename} | skills_found={len(resume_set)} | '
                      f'matched={len(matched)} | exp={exp_years}yr | '
                      f'edu_lvl={edu_level} | certs={len(certs)} | '
                      f'score={match_score} | cls={cls}')
                print(f'  -> matched_skills : {matched}')
                print(f'  -> top_skills     : {sorted(resume_set)[:12]}')
                print(f'  -> score_breakdown: skill={sd["skill_comp"]} '
                      f'exp={sd["exp_comp"]} edu={sd["edu_comp"]} '
                      f'prof={sd["prof_comp"]}')

                vector = [1 if n in resume_set else 0 for n,*_ in _SR]
                valid_indices.append(len(results))
                skill_vectors.append(vector)

                results.append({
                    'filename':        file.filename,
                    'name':            identity['name'],
                    'email':           identity['email'],
                    'phone':           identity['phone'],
                    'linkedin':        identity['linkedin'],
                    'github':          identity['github'],
                    'match_score':     match_score,
                    'skill_score':     skill_score,
                    'project_score':   proj_score,
                    'exp_score':       exp_score,
                    'candidate_exp':   exp_years,
                    'has_edu':         has_edu,
                    'edu_level':       edu_level,
                    'edu_label':       edu_labels.get(edu_level, 'Unknown'),
                    'certifications':  certs,
                    'matched_skills':  matched,
                    'missing_skills':  missing,
                    'recommendations': build_recommendations(missing),
                    'total_jd_skills': total_jd,
                    'skills_list':     sorted(resume_set),
                    'skill_confidence':skill_conf,
                    'cluster':         cluster,
                    'classification':  cls,
                    'similar_candidates': [],
                    'is_duplicate':    False,
                    'score_breakdown': sd,
                })
            except Exception as e:
                app.logger.warning(f'[MATCH] scoring failed {file.filename}: {e}')
                results.append(_fallback(file.filename, jd_skills, total_jd))

        n_valid = len(valid_indices)
        print(f'[DEBUG] phase1 done: {len(results)} results, {n_valid} valid for ML')

        # ── ML phase ──────────────────────────────────────────────────────────
        final_rules = []

        # KMeans
        if n_valid >= 2:
            try:
                va = np.array(skill_vectors, dtype=float)
                if not np.all(va.sum(axis=1) == 0):
                    k = min(4, n_valid)
                    KMeans(n_clusters=k, random_state=42, n_init='auto').fit_predict(va)
                app.logger.info('[KMEANS] done')
            except Exception as e:
                app.logger.warning(f'[KMEANS] {e}')

        # Apriori
        if apriori and n_valid >= 3:
            try:
                all_s = [results[ri]['skills_list'] for ri in valid_indices]
                cnts  = Counter(s for lst in all_s for s in lst)
                uniq  = sorted(s for s,_ in cnts.most_common(25))
                if len(uniq) >= 2:
                    df   = pd.DataFrame([{s:(s in c) for s in uniq} for c in all_s], dtype=bool)
                    freq = apriori(df, min_support=0.3, use_colnames=True)
                    if not freq.empty:
                        kw = dict(metric='confidence', min_threshold=0.7)
                        if _AR_NUM: kw['num_itemsets'] = len(df)
                        rules = association_rules(freq, **kw)
                        for _, row in rules.iterrows():
                            a = list(row['antecedents'])[0]
                            c2 = list(row['consequents'])[0]
                            final_rules.append(f"{a} \u2192 {c2} ({round(row['confidence']*100)}%)")
                app.logger.info(f'[APRIORI] {len(final_rules)} rules')
            except Exception as e:
                app.logger.warning(f'[APRIORI] {e}')

        # Cosine similarity + duplicate detection
        if n_valid >= 2:
            try:
                va  = np.array(skill_vectors, dtype=float)
                sim = cosine_similarity(va)
                for vi, ri in enumerate(valid_indices):
                    order = sim[vi].argsort()[::-1]
                    similar = []
                    for si in order:
                        if si == vi: continue
                        if sim[vi][si] > 0.35:
                            similar.append(results[valid_indices[si]]['filename'])
                        if len(similar) >= 2: break
                    results[ri]['similar_candidates'] = similar
                    best_other = next((si for si in order if si != vi), None)
                    results[ri]['is_duplicate'] = bool(
                        best_other is not None and sim[vi][best_other] > 0.92
                    )
                app.logger.info('[COSINE] done')
            except Exception as e:
                app.logger.warning(f'[COSINE] {e}')

        # Final fallback
        if not results:
            for file in files:
                if file.filename:
                    results.append(_fallback(file.filename, jd_skills, total_jd))

        results.sort(key=lambda r: r['match_score'], reverse=True)
        print(f'[DEBUG] returning {len(results)} results')
        return jsonify({
            'results':           results,
            'job_description':   job_desc,
            'association_rules': final_rules[:5],
        })

    except Exception as e:
        print(f'[ERROR] {e}')
        app.logger.exception(f'[MATCH] {e}')
        return jsonify({'results': [], 'association_rules': [], 'error': str(e)}), 200


# ── Error handlers ────────────────────────────────────────────────────────────
@app.errorhandler(404)
def not_found(e): return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(405)
def method_not_allowed(e): return jsonify({'error': 'Method not allowed'}), 405

@app.errorhandler(413)
def too_large(e): return jsonify({'error': 'File too large (max 16 MB)'}), 413

@app.errorhandler(500)
def server_error(e): return jsonify({'error': f'Internal server error: {e}'}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)
