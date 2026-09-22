from django.shortcuts import render
from django.http import JsonResponse

import re
import requests
import os
import json
import tempfile
import pytesseract

from pdf2image import convert_from_path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =========================
# PATHS / CONFIG
# =========================

TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
POPPLER_PATH = r"E:\poppler\poppler-26.09.0\Library\bin"

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
OLLAMA_MODEL = "llama3.2"

pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH


# =========================
# TEXT MATCHING HELPERS
# =========================

def contains_term(text, term):
    """
    Checks a complete word/phrase.
    Prevents Java from matching JavaScript.
    """
    pattern = r"(?<!\w)" + re.escape(term.lower()) + r"(?!\w)"
    return re.search(pattern, text.lower()) is not None


def contains_any(text, terms):
    for term in terms:
        if contains_term(text, term):
            return True
    return False


# =========================
# OLLAMA JOB REQUIREMENT EXTRACTION
# =========================

def extract_job_requirements(job_description):
    """
    Extract only requirements explicitly mentioned in the JD.
    Uses Python matching instead of relying on Ollama.
    """

    if not job_description.strip():
        return []

    # name, aliases, category
    known_requirements = [

        # Technical
        {
            "name": "Python",
            "aliases": ["python"],
            "category": "technical"
        },
        {
            "name": "Django",
            "aliases": ["django"],
            "category": "technical"
        },
        {
            "name": "Java",
            "aliases": ["java"],
            "category": "technical"
        },
        {
            "name": "C++",
            "aliases": ["c++", "cpp"],
            "category": "technical"
        },
        {
            "name": "JavaScript",
            "aliases": ["javascript", "js"],
            "category": "technical"
        },
        {
            "name": "HTML",
            "aliases": ["html"],
            "category": "technical"
        },
        {
            "name": "CSS",
            "aliases": ["css"],
            "category": "technical"
        },
        {
            "name": "React",
            "aliases": ["react", "react.js"],
            "category": "technical"
        },
        {
            "name": "Flask",
            "aliases": ["flask"],
            "category": "technical"
        },
        {
            "name": "SQL",
            "aliases": ["sql"],
            "category": "technical"
        },
        {
            "name": "MySQL",
            "aliases": ["mysql"],
            "category": "technical"
        },
        {
            "name": "PostgreSQL",
            "aliases": ["postgresql", "postgres"],
            "category": "technical"
        },
        {
            "name": "MongoDB",
            "aliases": ["mongodb", "mongo db"],
            "category": "technical"
        },
        {
            "name": "Git",
            "aliases": ["git"],
            "category": "technical"
        },
        {
            "name": "GitHub",
            "aliases": ["github", "git hub"],
            "category": "technical"
        },
        {
            "name": "GitLab",
            "aliases": ["gitlab", "git lab"],
            "category": "technical"
        },
        {
            "name": "Machine Learning",
            "aliases": ["machine learning", "ml"],
            "category": "technical"
        },
        {
            "name": "Deep Learning",
            "aliases": ["deep learning"],
            "category": "technical"
        },
        {
            "name": "Artificial Intelligence",
            "aliases": [
                "artificial intelligence",
                "ai"
            ],
            "category": "technical"
        },

        # Core CS
        {
            "name": "Object Oriented Programming",
            "aliases": [
                "object oriented programming",
                "object-oriented programming",
                "oop"
            ],
            "category": "core"
        },
        {
            "name": "Data Structures",
            "aliases": [
                "data structures",
                "data structure"
            ],
            "category": "core"
        },
        {
            "name": "Algorithms",
            "aliases": ["algorithms", "algorithm"],
            "category": "core"
        },
        {
            "name": "Problem Solving",
            "aliases": [
                "problem solving",
                "problem-solving"
            ],
            "category": "core"
        },
        {
            "name": "Debugging",
            "aliases": ["debugging", "debug"],
            "category": "core"
        },

        # Other
        {
            "name": "Communication",
            "aliases": [
                "communication",
                "communication skills"
            ],
            "category": "other"
        },
        {
            "name": "Teamwork",
            "aliases": [
                "teamwork",
                "team work"
            ],
            "category": "other"
        },
        {
            "name": "Leadership",
            "aliases": ["leadership"],
            "category": "other"
        }
    ]

    requirements = []

    for requirement in known_requirements:

        if contains_any(
            job_description.lower(),
            requirement["aliases"]
        ):

            requirements.append({
                "name": requirement["name"],
                "aliases": requirement["aliases"],
                "category": requirement["category"]
            })

    return requirements
def home(request):

    context = {}

    if request.method == "POST":

        uploaded_file = request.FILES.get("resume")
        job_description = request.POST.get(
            "job_description",
            ""
        ).strip()

        # -------------------------
        # Resume validation
        # -------------------------

        if not uploaded_file:
            context["error"] = "Please upload a resume PDF."
            return render(
                request,
                "index.html",
                context
            )

        if not uploaded_file.name.lower().endswith(".pdf"):
            context["error"] = "Only PDF files are allowed."
            return render(
                request,
                "index.html",
                context
            )

        # -------------------------
        # Temporary PDF
        # -------------------------

        temp_pdf = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        )

        pdf_path = temp_pdf.name
        temp_pdf.close()

        try:

            with open(pdf_path, "wb") as destination:
                for chunk in uploaded_file.chunks():
                    destination.write(chunk)

            # -------------------------
            # PDF → image
            # -------------------------

            pages = convert_from_path(
                pdf_path,
                poppler_path=POPPLER_PATH
            )

            # -------------------------
            # OCR
            # -------------------------

            extracted_text = ""

            for page in pages:
                text = pytesseract.image_to_string(page)
                extracted_text += "\n" + text

        except Exception as e:

            context["error"] = (
                "Could not process the PDF. "
                f"Error: {str(e)}"
            )

            return render(
                request,
                "index.html",
                context
            )

        finally:

            if os.path.exists(pdf_path):
                try:
                    os.remove(pdf_path)
                except Exception:
                    pass

        # -------------------------
        # OCR validation
        # -------------------------

        resume_text = extracted_text.strip()

        if not resume_text:

            context["error"] = (
                "No readable text was found in the PDF."
            )

            return render(
                request,
                "index.html",
                context
            )

        resume_lower = resume_text.lower()
        job_lower = job_description.lower()

        # =========================
        # RESUME SECTIONS
        # =========================

        sections = {
            "Skills": contains_any(
                resume_lower,
                [
                    "skills",
                    "technical skills",
                    "skill set"
                ]
            ),

            "Education": contains_any(
                resume_lower,
                [
                    "education",
                    "academic background",
                    "qualification"
                ]
            ),

            "Experience": contains_any(
                resume_lower,
                [
                    "experience",
                    "work experience",
                    "internship",
                    "internships"
                ]
            ),

            "Projects": contains_any(
                resume_lower,
                [
                    "projects",
                    "project"
                ]
            ),

            "Certifications": contains_any(
                resume_lower,
                [
                    "certifications",
                    "certification",
                    "certificates",
                    "certificate"
                ]
            )
        }

        # =========================
        # RESUME SCORE
        # =========================

        resume_score = sum(
            20 for value in sections.values()
            if value
        )

        missing_sections = [
            name
            for name, found in sections.items()
            if not found
        ]

        # =========================
        # RESUME SKILLS
        # =========================

        skill_database = {

            "Programming": [
                "Python",
                "Java",
                "C++",
                "C",
                "JavaScript",
                "TypeScript"
            ],

            "Web": [
                "HTML",
                "CSS",
                "JavaScript",
                "React",
                "Django",
                "Flask"
            ],

            "Database": [
                "SQL",
                "MySQL",
                "PostgreSQL",
                "MongoDB"
            ],

            "Git": [
                "Git",
                "GitHub",
                "GitLab"
            ],

            "AI/ML": [
                "Machine Learning",
                "Deep Learning",
                "Artificial Intelligence",
                "NLP",
                "TensorFlow",
                "PyTorch",
                "Scikit-learn"
            ],

            "Data": [
                "NumPy",
                "Pandas",
                "Matplotlib",
                "Power BI",
                "Excel"
            ],

            "Core CS": [
                "Object Oriented Programming",
                "OOP",
                "Data Structures",
                "Algorithms",
                "Problem Solving",
                "Debugging"
            ],

            "Other": [
                "Communication",
                "Teamwork",
                "Leadership"
            ]
        }

        skills_found = []

        for category, skills in skill_database.items():

            for skill in skills:

                if contains_term(
                    resume_lower,
                    skill
                ):

                    if skill not in skills_found:
                        skills_found.append(skill)

        # =========================
        # COMMON MISSING SKILLS
        # =========================

        common_skills = [
            "Python",
            "SQL",
            "Git",
            "GitHub",
            "Django",
            "HTML",
            "CSS",
            "JavaScript"
        ]

        missing_skills = [
            skill
            for skill in common_skills
            if not contains_term(
                resume_lower,
                skill
            )
        ]

        # =========================
        # GENERAL JOB MATCHING
        # =========================

        matched_keywords = []
        missing_keywords = []

        technical_matched = []
        technical_missing = []

        core_matched = []
        core_missing = []

        other_matched = []
        other_missing = []

        if job_description:

            requirements = extract_job_requirements(
                job_description
            )

            for requirement in requirements:

                name = requirement["name"]
                aliases = requirement["aliases"]
                category = requirement["category"]

                if contains_any(
                    resume_lower,
                    aliases
                ):

                    matched_keywords.append(name)

                    if category == "technical":
                        technical_matched.append(name)

                    elif category == "core":
                        core_matched.append(name)

                    else:
                        other_matched.append(name)

                else:

                    missing_keywords.append(name)

                    if category == "technical":
                        technical_missing.append(name)

                    elif category == "core":
                        core_missing.append(name)

                    else:
                        other_missing.append(name)

            # -------------------------
            # Match percentage
            # -------------------------

            total_requirements = len(
                requirements
            )

            total_matched = len(
                matched_keywords
            )

            if total_requirements > 0:

                match_percentage = round(
                    (
                        total_matched
                        / total_requirements
                    ) * 100
                )

            else:
                match_percentage = 0

            # -------------------------
            # TF-IDF similarity
            # -------------------------

            try:

                vectorizer = TfidfVectorizer()

                tfidf_matrix = vectorizer.fit_transform(
                    [
                        resume_lower,
                        job_lower
                    ]
                )

                similarity_score = round(
                    cosine_similarity(
                        tfidf_matrix[0:1],
                        tfidf_matrix[1:2]
                    )[0][0] * 100
                )

            except Exception:

                similarity_score = 0

        else:

            match_percentage = 0
            similarity_score = 0

        # =========================
        # STRENGTHS
        # =========================

        strengths = []

        if sections["Projects"]:
            strengths.append(
                "Projects section is present, "
                "which helps demonstrate practical experience."
            )

        if sections["Experience"]:
            strengths.append(
                "Experience or internship information is included."
            )

        if sections["Education"]:
            strengths.append(
                "Education section is clearly represented."
            )

        if "Python" in skills_found:
            strengths.append(
                "Python is present as a technical skill."
            )

        if len(skills_found) >= 5:
            strengths.append(
                "Resume contains a useful range of technical skills."
            )

        # =========================
        # IMPROVEMENT AREAS
        # =========================

        improvement_areas = []

        if missing_sections:
            improvement_areas.append(
                "Consider adding or improving these resume "
                "sections: "
                + ", ".join(missing_sections)
            )

        if missing_skills:
            improvement_areas.append(
                "Add more relevant technical skills "
                "that you genuinely know."
            )

            improvement_areas.append(
                "Consider learning or highlighting relevant "
                "skills such as: "
                + ", ".join(missing_skills)
            )

        if job_description and missing_keywords:

            improvement_areas.append(
                "For this specific job, address missing "
                "requirements where you genuinely have "
                "the skill or experience."
            )

        if core_missing:

            improvement_areas.append(
                "Strengthen your Core Computer Science "
                "knowledge in: "
                + ", ".join(core_missing)
            )

        if job_description:

            improvement_areas.append(
                "Tailor your resume wording to the job "
                "description by accurately describing "
                "relevant projects, skills, and experience."
            )

        # =========================
        # RECOMMENDATIONS
        # =========================

        recommendations = []

        if missing_skills:

            recommendations.append(
                "Consider adding these commonly requested "
                "skills: "
                + ", ".join(missing_skills)
            )

        if missing_sections:

            recommendations.append(
                "Consider adding missing resume sections: "
                + ", ".join(missing_sections)
            )

        if job_description and missing_keywords:

            recommendations.append(
                "For this job, consider highlighting these "
                "missing requirements: "
                + ", ".join(missing_keywords)
            )

        if matched_keywords:

            recommendations.append(
                "Strengthen your matched skills by showing "
                "projects, internship work, or measurable "
                "achievements."
            )

        if core_missing:

            recommendations.append(
                "Improve your Core CS preparation in: "
                + ", ".join(core_missing)
            )

        if (
            job_description
            and similarity_score < 40
        ):

            recommendations.append(
                "Your resume wording has relatively low "
                "textual similarity to this job description. "
                "Consider tailoring relevant project and "
                "experience descriptions using accurate "
                "job-related terminology."
            )

        # =========================
        # REMOVE DUPLICATES
        # =========================

        strengths = list(dict.fromkeys(strengths))
        improvement_areas = list(
            dict.fromkeys(improvement_areas)
        )
        recommendations = list(
            dict.fromkeys(recommendations)
        )

        # =========================
        # TEMPLATE CONTEXT
        # =========================

        context.update({

            "resume_text": resume_text,

            "job_description": job_description,

            "resume_score": resume_score,

            "sections": sections,

            "missing_sections": missing_sections,

            "skills_found": skills_found,

            "missing_skills": missing_skills,

            "matched_keywords": matched_keywords,

            "missing_keywords": missing_keywords,

            "match_percentage": match_percentage,

            "technical_matched": technical_matched,

            "technical_missing": technical_missing,

            "core_matched": core_matched,

            "core_missing": core_missing,

            "other_matched": other_matched,

            "other_missing": other_missing,

            "similarity_score": similarity_score,

            "strengths": strengths,

            "improvement_areas": improvement_areas,

            "recommendations": recommendations
        })

    return render(
        request,
        "index.html",
        context
    )


# =========================
# AI RESUME REWRITER
# =========================

def rewrite_resume(request):

    if request.method != "POST":

        return JsonResponse({
            "success": False,
            "error": "POST request required."
        })

    rewrite_text = request.POST.get(
        "rewrite_text",
        ""
    ).strip()

    job_description = request.POST.get(
        "job_description",
        ""
    ).strip()

    if not rewrite_text:

        return JsonResponse({
            "success": False,
            "error": "Please enter resume content."
        })

    prompt = f"""
Rewrite the following resume content professionally.

Rules:
- Keep the original meaning.
- Make it concise and professional.
- Make it suitable for a Software Engineering resume.
- Focus on action and impact.
- Do NOT invent experience.
- Do NOT invent technologies.
- Do NOT invent numbers or achievements.
- Return only the improved sentence or bullet point.

Resume content:
{rewrite_text}

Job description:
{job_description}
"""

    try:

        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.2,
                    "num_predict": 200
                }
            },
            timeout=120
        )

        response.raise_for_status()

        data = response.json()

        improved_text = data.get(
            "response",
            ""
        ).strip()

        if not improved_text:

            return JsonResponse({
                "success": False,
                "error": "AI returned an empty response."
            })

        return JsonResponse({
            "success": True,
            "rewritten_text": improved_text
        })

    except requests.exceptions.ConnectionError:

        return JsonResponse({
            "success": False,
            "error": (
                "Could not connect to Ollama. "
                "Make sure Ollama is running."
            )
        })

    except requests.exceptions.Timeout:

        return JsonResponse({
            "success": False,
            "error": "Ollama request timed out."
        })

    except requests.exceptions.RequestException as e:

        return JsonResponse({
            "success": False,
            "error": f"Ollama request failed: {str(e)}"
        })

    except Exception as e:

        return JsonResponse({
            "success": False,
            "error": f"Unexpected error: {str(e)}"
        })