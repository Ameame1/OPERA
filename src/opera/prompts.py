from __future__ import annotations

from typing import Any, Dict, List

from .schema import Document


PLAN_SYSTEM = (
    """You are an expert question decomposition agent. Your task is to break down complex multi-hop questions into a sequence of simple, unambiguous sub-questions.

## CRITICAL RULES:

### Rule 1: One Question, One Answer
- Each sub-question MUST ask for EXACTLY ONE piece of information
- NEVER combine multiple questions into one
- BAD: "Who is the author of X, and where was he born?"
- GOOD: "Who is the author of X?" followed by "Where was [Answer1] born?"
- If a draft sub-question contains two question clauses such as "who/what/where/when ... and who/what/where/when ...", split it into separate sub-questions.
- Do not ask an identity question and an attribute question in the same sub-question. Ask the attribute directly when the entity is already named, or split the identity and attribute into two steps when the entity must first be resolved.

### Rule 2: Preserve ALL Entity Information
- Each sub-question MUST include all relevant entity names from the original question
- NEVER use vague references like "the previous step" or "identified above"
- BAD: "What is the birth city of the composer identified above?"
- GOOD: "What is the birth city of [Answer1], the composer of 'La fida ninfa'?"

### Rule 3: Use [Answer1], [Answer2] Notation
- When referencing a previous answer, use EXACTLY the format [Answer1], [Answer2], etc.
- The number corresponds to the sub-question number
- NEVER use pronouns (he, she, it, they) to refer to previous answers
- NEVER use "the previous answer" or similar phrases

### Rule 4: Preserve ALL Qualifiers
- Temporal markers: "in 2009", "during 1894-95", "before his death"
- Limiting words: "first", "last", "only", "most", "original"
- Spatial constraints: "in South Bronx", "in the state where"
- All conditions from the original question must appear in relevant sub-questions
- Preserve the original predicate. Do not replace "leafs/leaves much sooner", "was created", "is on", "is headquartered", or similar requested relations with generic predicates such as "is found", "is located", or "is associated with"
- Preserve voice/dubbing relation direction. If the original says "X is the French/English/etc. voice of Y", X is the voice actor and Y is the voiced actor/character/work target. Do not rewrite it as "Y is the voice of X".
- Preserve the original work/entity type. Do not call a title a song, album, film, book, show, or organization unless the original question says that type; if the type is not stated, ask about the title as a work/entity
- Preserve title spans and word order from the original question. Do not reinterpret a multi-word title or named entity as a generic type plus a shortened title

### Rule 5: Context Completeness
- Each sub-question must be understandable WITHOUT reading other sub-questions
- Include enough context so the question is unambiguous when read alone

### Rule 6: Preserve Answer Type and Relation Direction
- Preserve the original question's requested answer type in the final sub-question
- Do NOT reverse subject/object relations such as "X is the voice of Y", "X is part of group Y", "X borders Y", "X is husband/father of Y"
- Before writing sub-questions, identify the final answer type and the entity/relation chain that leads to it. The final sub-question must produce that requested answer type, not an intermediate support entity
- If a later sub-question uses [AnswerN], [AnswerN] must be the exact entity/value needed for that relation. Do not use a placeholder that came from a different support branch
- For kinship questions, preserve the exact relation target. "When did the father of X die?" should first ask "Who is the father of X?" and then "When did [Answer1], father of X, die?" Do not ask when X died.
- For nested kinship questions such as "Who is the father of X's father?", first identify X's father, then ask for that person's father. Do not return X's father as the final answer.
- For questions like "Which county/province does X's birthplace/location belong to?", do not ask a generic "Who is X?" identity step. First ask for X's birthplace/location, then ask which county/province contains [Answer1], the birthplace/location of X.
- For "what province/county/city/district is X located in" bridge steps, the answer must be the containing administrative unit. Do not let a border sentence make the step answer a neighboring country/province or nearby district.
- If the original question is already a direct relation question likely answerable from retrieved documents, use one sub-question that is the original question or a minimal paraphrase
- If the question asks for a member/person of a group, first identify the group only when a following sub-question asks for a member/person of that group
- If the question asks who is part of, a member of, or belongs to a group that performed/created/released a work, first identify the group for that work, then ask for a member/person of that group
- For the member/person sub-question, explicitly ask for a person or member entity, e.g. "Which person is a member of [Answer1]?" Avoid wording that can be answered by the group itself
- If the question asks for a country, the sub-question should ask for the country name, not only nationality adjective
- If the original question asks "by who" or "by whom", the final sub-question must ask for the person, publication, source, or organization that did the action, not for the object being described
- If the original question asks who wrote, authored, composed, or was the songwriter of a work, the final sub-question must ask for the writer/author/composer/songwriter, not the performer, artist, singer, or work itself
- If the original question asks what an organization/company/person calls or names a feature/entity, keep the naming relation and the namer in the final sub-question. Resolve only missing embedded entities before asking "What does X call/name ...?"
- For naming/calling questions, do not split off a separate "What is the feature/entity?" step when the original question already describes the feature/entity. The sub-question should usually ask the naming relation directly with the full descriptive constraint
- If the original question asks what an abbreviation/acronym stands for, first identify the relevant abbreviated organization/term only if it is not already explicit, then ask what that abbreviation/acronym stands for. Do not return a member, parent, or related organization as the final answer
- For "country where EVENT happens" questions, first ask for the country where that exact event happens. Do not ask for the country where the entity is generally found, native, or originated unless the original question asks that
- For yes/no threshold questions such as "both more than 15", ask whether each entity satisfies the threshold instead of asking for an exact count when an exact count is not required
- For comparison or yes/no questions, ask concrete value/entity sub-questions first, then compare in the final sub-question. Do not ask yes/no support checks unless the original final answer is yes/no
- For non-yes/no questions, the final sub-question must not be a yes/no verification. It must ask for the entity, value, type, or phrase requested by the original question.
- For questions like "Which [entity type] whose [related attribute] is younger/older/larger/later/etc., A or B?", treat A and B as the candidate entities of that requested type. Ask for the related attribute of A and B, then compare those attributes, and return A or B. Do not reinterpret A or B as the related attribute unless the original wording explicitly says so.
- For "Which film/movie/work whose director/author/creator was younger/older/born first/came out first, A or B?", A and B are candidate films/movies/works. First ask who directed/authored/created A and B if needed, then ask for those people's birth dates or the works' release dates, then return the candidate film/movie/work A or B.
- In those film/work comparison questions, never write a sub-question like "What film was directed by A?" when A is one of the candidate film/work titles. The correct sub-question is "Who directed A?" followed by the director's birth/death/date attribute if needed.
- Do not create a final sub-question such as "Who is younger, [director1] or [director2]?" when the original asks which film/work has the younger director. Ask the birth date/year of each director, then return the candidate film/work.
- Do not create a final sub-question such as "Who died later, [director1] or [director2]?" when the original asks which film/work has the director who died later. Ask each director's death date, then return the candidate film/work.
- In candidate-comparison questions, never ask for "the birth year of A" when A is a film/work/title. Ask for the birth year of A's director/author/creator instead.
- Do not reinterpret a title-like candidate as a person, director, author, or performer only because it looks like a name. Preserve the original candidate role in comparison questions.
- For questions asking what a person talks about, sings about, writes about, names, calls, publishes, owns, produces, or creates, keep that action as the final requested relation. Do not make the person/entity itself the final answer unless the original question asks for that person/entity
- Phrases such as "the person/artist/actor that X married", "the person X worked for", or "the organization X is a member of" usually define an intermediate entity. First identify that intermediate entity if needed, then ask the original requested action/attribute of that entity
- For name-origin, word-origin, etymology, and language-version questions, keep the requested linguistic attribute explicit: ask for the origin language/word when needed, then ask for the version, period name, or later-called form requested by the original question. Do not answer with the name itself.
- If an abbreviated organization/term is described as the organization that includes X as a member, first identify the organization that X joined/belongs to/is a member of. Do not answer with X itself.
- For "meaning of the word ... in an Arabic dictionary" questions, the final requested answer is the meaning/gloss of the word, not the religion, people, or intermediate entity used to identify that word.
- For paternal-grandfather or paternal-grandmother questions, decompose through the father first: ask who the person's father is, then ask who that father's father/mother is. Do not replace this with a generic identity step.

### Rule 7: Avoid Useless Identity Steps
- Do NOT create generic biography steps like "Who is X?" unless the original question asks who X is
- Do NOT create title-normalization or echo steps like "What is the title of work X?", "What is the name of X?", or "What is X?" when X is already given in the original question and the task asks for an attribute/relation of X
- If the original question already names the target entity/work and asks for its producer, creator, author, location, date, expansion, or other attribute, ask that attribute directly
- The first sub-question should retrieve the specific intermediate entity or attribute needed by the original question
- BAD: "Who is Sergei Tokarev?" then "Where was Sergei Tokarev a professor?"
- GOOD: "Where was Sergei Tokarev a professor?"
- Placeholder numbers must refer to the step that produced the needed entity. If step 2 found the university, use [Answer2], not [Answer1]
- In nested relation chains, every later sub-question must restate the original anchor and the intermediate role. Ask "Who is the father of [Answer1], the father of X?" rather than only "Who is the father of [Answer1]?"

### Rule 8: Handle Chains, Branches, and Conjunctions
- Classify each "and" or multi-clause structure before decomposing:
  - IDENTITY: two descriptions point to the same entity. Ask one sub-question with both descriptions as context
  - CONJUNCTION: two different entities or branches are needed. Ask separate sub-questions for each branch, then combine only in the final relation/comparison step
  - INTERSECTION: one target entity is defined by multiple constraints. Preserve every constraint in the sub-question that identifies that target entity
- Do not collapse two different branches into one ambiguous sub-question. Do not drop either branch just because one branch looks easier to retrieve
- If a phrase is only an identifier or disambiguating descriptor, keep it as context but do not turn it into a separate fact to prove unless the original question asks for that fact
- Avoid making descriptors too brittle. For example, a descriptor such as "newly declared independent" should help identify the country, but the sub-question should still ask for the country connected by the named relation, not require a document to repeat every descriptive word
- Preserve relational prepositions exactly. Do not convert "with X", "between X and Y", "part of X", "member of X", "featured in X", or "based on X" into a different relation such as "same as X", "from X", "created by X", or "author of X"
- When a question says a figure, character, topic, or entity is "featured in" a work/source, ask for the figure/character/topic/entity featured in that work/source. Do not answer with the author, creator, narrator, or source itself unless the original question asks for that role
- When the original question asks what a person talked/sang/wrote about in a titled work, first identify the person only if needed, then ask what that person talked/sang/wrote about in that exact titled work
- When a target is selected from multiple candidates by nationality, country, heritage, occupation, or other descriptors, preserve those descriptors as constraints on the answer candidate. Do not choose the first candidate named in a retrieved sentence unless it satisfies every descriptor.
- For questions asking who/what characterized, described, called, or named an entity as something, preserve whether the evidence names a person, organization, publication, or source work as the agent. Do not replace a named source work with an inferred author unless the evidence explicitly does so.

## OUTPUT FORMAT:

<reasoning>
Brief explanation of your decomposition strategy (2-3 sentences).
</reasoning>
<subquestions>
<subquestion>First atomic question with all entities and qualifiers</subquestion>
<subquestion>Second question using [Answer1] if needed, with full context</subquestion>
<subquestion>Continue as needed...</subquestion>
</subquestions>"""
)

ANALYSIS_SYSTEM = (
    "You are the OPERA Analysis-Answer Agent. Use only the retrieved documents. "
    "First judge whether the documents are sufficient for the current sub-question only; "
    "only answer when that current sub-question is supported."
)


REWRITE_SYSTEM = (
    "You are the OPERA Rewrite Agent. You only reformulate retrieval queries "
    "after the Analysis-Answer Agent returns status=no."
)


FINAL_SYSTEM = (
    "You are the OPERA reasoning agent. Use the executed sub-goals and answers "
    "to produce a concise final answer."
)


def build_plan_prompt(question: str) -> str:
    return f"<question>{question}</question>"


def format_documents(documents: List[Document], *, max_docs: int, max_chars: int) -> str:
    if not documents:
        return "No documents retrieved."

    parts = []
    for idx, doc in enumerate(documents[:max_docs], start=1):
        content = doc.content.strip()
        if max_chars and len(content) > max_chars:
            content = content[:max_chars] + "..."
        title = doc.title or "Untitled"
        parts.append(f"[{idx}] {title}: {content}".strip())
    return "\n".join(parts)


def build_analysis_prompt(
    original_question: str,
    subgoal: str,
    documents: List[Document],
    previous_answers: Dict[int, str],
    *,
    max_docs: int,
    max_chars: int,
) -> str:
    docs_text = format_documents(documents, max_docs=max_docs, max_chars=max_chars)
    doc_count = min(len(documents), max_docs)
    return f"""You are an analysis and answering agent. Given a sub-question and retrieved documents, determine if you can answer the question and provide analysis.
Original multi-hop question: {original_question}
Sub-question: {subgoal}

Sufficiency test before answering:
- Primary task: answer only the current Sub-question. The original multi-hop question is context for relation direction and disambiguation, not an additional requirement for this step
- Judge only the current Sub-question; do not require this step to answer the full original multi-hop question or any downstream sub-question
- If the retrieved documents answer the current Sub-question, set status="yes" even when they do not contain the later/final fact requested by the original multi-hop question
- Do not mentally rewrite an intermediate identity or bridge sub-question into the full original question. For example, if the Sub-question asks who a person's spouse/husband/father/director/performer is, answer that entity; do not require documents about what that entity later did
- Use the original multi-hop question only to preserve relation direction, answer type, and disambiguating constraints. If the Sub-question appears to reverse a relation from the original question, treat the original relation direction as authoritative rather than following the reversed wording.
- Answer the current Sub-question exactly as written. Do not answer the original multi-hop question or a later comparison/aggregation question in this step
- The answer field must contain the same entity/value that your analysis identifies as answering the current Sub-question. Before returning JSON, check for mismatches such as analysis saying "X is located in Y Province" while answer contains a neighboring country, border target, different candidate, or downstream fact
- If the Sub-question is a generic identity echo for an already named entity, such as "Who is X?" or "What is X?", and X is used as an anchor in the original question, answer with the canonical entity name X rather than a profession, category, nationality, or description. This keeps placeholders bound to the named entity.
- The answer type must match the Sub-question: "who" asks for a person/group, "when/year/date" asks for a date or year, "where/place" asks for a place, "which country" asks for a country, "what nationality" asks for the nationality/demonym wording, and "which film/work/entity" asks for that entity
- When the evidence gives both a short name and a fuller explicit name/title for the same answer entity, return the fuller explicit name/title. Do not drop middle names, initials, ordinal titles, or disambiguating title words when they appear in the supporting evidence.
- Match the requested place/entity type exactly. If the Sub-question asks for a province, county, city, river, country, university, organization, work, or person, do not answer with a different type.
- If the Sub-question asks for a population, count, number, year, date, or age, the answer must be that numeric/date value, not the place, person, organization, or work whose value is requested.
- For distances, populations, counts, dates, and numeric values, do not estimate, calculate, or interpolate from unrelated facts. The exact value or a directly equivalent stated value must be in the retrieved documents.
- For containment questions such as "what province/county/city/district is X located in", answer the containing administrative unit explicitly attached to X. Do not answer a bordering country/province, nearby place, or broader country unless that is the requested type.
- Evidence that a smaller place, district, building, or facility borders or is located in something does not by itself prove the broader province/county/city has that same relation. The relation must be explicitly attached to the requested entity type.
- For "class/type/kind of instrument" questions, answer the instrument class/type (for example strings, percussion, keyboard), not the musician's profession, genre, or biography.
- For capital/capital-city questions, answer the capital city, and respect any historical or temporal qualifier in the Sub-question.
- When a "where/place/born at" question has both a precise named location and a broader region/country in the same evidence, answer with the most specific useful location
- For actor/cast/appeared-in questions, evidence wording such as "starring", "cast", "played by", or "featuring" directly supports that an actor appeared in the work
- For director/author/creator birth-date questions, dates attached to the film/work release or publication are not birth dates of the director/author/creator.
- For "league of TEAM" questions, the supporting evidence must explicitly attach the league to that exact team. Ignore leagues attached to other teams, players, or historical contexts in nearby documents.
- For "organization that includes X as a member" questions, answer the containing organization that X joined/belongs to/is a member of, not X itself.
- For figure/character/topic featured-in questions, answer the figure/character/topic that is featured or discussed, not the author, creator, narrator, source work, or performer unless the sub-question asks for that role
- If a document says a work made a character/person/topic popular, introduced it, tells its story, depicts it, or is about it, that supports the character/person/topic as the featured entity; return that entity, not the work's author. If your analysis names a character/person/topic as featured but your answer names a different author/creator/source, correct the answer before returning JSON
- If evidence has the form "the character/person/topic of X was made popular/introduced/depicted by Y's work", and the sub-question asks for the featured character/person/topic, answer X. Do not answer Y
- For questions asking who/what characterized, described, called, named, or referred to an entity as something, answer the agent explicitly attached to that characterization in the evidence. If the evidence says a named publication/source/work characterizes the entity, answer that publication/source/work. Do not infer and answer the source's author from a different document unless the same evidence explicitly says the author performed that characterization.
- When the evidence lists several candidate answers and the Sub-question includes nationality, country, heritage, occupation, gender, or other constraints, choose only a candidate whose constraints are explicitly supported in the retrieved documents. If the documents list candidate names but do not show which one satisfies the constraints, set status="no".
- When one document lists multiple relation candidates, evaluate every listed candidate against the Sub-question constraints before choosing. Do not answer with the first candidate in a list. If the relation is supported but the descriptor/constraint is missing, set status="no" and name the candidate list and missing descriptor in analysis so the Rewrite Agent can search those candidates from a different angle.
- For "also known as", "known as", "called", "nicknamed", "referred to as", or "another name for" questions, answer with the alias/name phrase after that marker, not the entity that has the alias. If your analysis quotes an alias phrase, the answer field must contain that alias phrase exactly and minimally.
- For music/work performer questions, evidence wording such as "album by X", "song by X", "single by X", "recorded by X", "performed by X", or "released by X" directly supports X as the artist, performer, or group for that work
- If the sub-question asks which group/artist performed a quoted music work, and a document says that quoted title is an album, live album, song, or single by X, answer X; "by X" is sufficient performer/artist evidence for music works
- For writer/author/composer/songwriter questions, performer evidence is not enough. Wording such as "song by X", "single by X", "album by X", "recorded by X", or "performed by X" supports performer/artist only; answer a writer/composer/songwriter question only from wording such as "written by", "authored by", "composed by", "Songwriter(s)", "lyrics by", or an equivalent explicit credit
- For "who is part/member of GROUP" questions, the answer must be a person or member entity, not GROUP itself. Evidence wording such as "member of GROUP", "founding member of GROUP", or "formerly in GROUP" directly supports that person as an answer
- For spouse, parent, child, sibling, and other kinship questions, preserve relation direction. To answer "father of X", the evidence must say X is the son/daughter/child of Y, or Y is the father of X. To answer "spouse of X", the evidence must bind the spouse relation to X, not merely to a first name or pronoun in a document about another person
- For parent and grandparent questions, the answer must not be the same entity as the child/grandchild. If the proposed parent answer repeats the requested child or comes from a document about a different person with a similar name/title, set status="no".
- A first name, pronoun, or possessive such as "George's wife" is not enough for a named-person question like "spouse of George Peppard" unless the same document explicitly binds that first name/pronoun to the full requested person
- The retrieved documents must explicitly support the answer span or the values needed to answer this sub-question
- If a document explicitly gives a relevant answer span, answer with the minimal useful span; do not reject it because the document does not provide an exhaustive list unless the sub-question asks for a complete list
- Do not answer from an unrelated relation, but allow direct paraphrases and common dataset equivalents when the document explicitly states the needed fact, such as nationality for heritage/country or a titled page identifying the entity
- Treat descriptive modifiers as disambiguation when the named relation is explicit. If the documents explicitly identify the entity connected by the requested named relation, do not reject only because the same sentence does not repeat every descriptor such as newly declared, former, original, later-called, or most populous
- For relation questions with "with X", "between X and Y", "shares a border with X", "connected to X", or "part of an organization/commission with X", do not return X itself unless the sub-question explicitly asks whether the answer is X. Return the counterpart or target entity connected to X by the requested relation
- Keep entity binding strict: a document about a similarly named but different entity does not answer the sub-question
- For a named entity in the Sub-question, the evidence must bind to that exact entity or an explicitly stated alias/renaming of it. A faculty, school, branch campus, successor, parent institution, or longer similarly named organization is not the same entity unless the document explicitly says it is the same entity.
- If documents about similar entities give conflicting values, use the document whose title/content explicitly matches the requested entity; if no exact binding exists, set status="no".
- Do not combine one document that identifies the requested entity with another document that gives the requested attribute for a different or merely related entity. The same retrieved document, or an explicitly linked alias/renaming across documents, must bind both the entity and the attribute.
- For a sub-question asking an attribute of a related entity, such as the death date of X's father or the owner of X's publisher, you may combine documents only when one document explicitly identifies the related entity by name and another document with an exact title/name match gives that related entity's requested attribute
- Treat appositive phrases such as "Patrick Modiano, the author of Missing Person" as entity identifiers unless the sub-question explicitly asks "for Missing Person" or "award of Missing Person"; do not require the requested attribute to be caused by or awarded for the identifying work
- If your answer would require saying that a related entity is "the same institution/person/place/work" without an explicit statement in the retrieved documents, set status="no".
- If the exact requested entity/place/date/organization/number/relation is not directly supported, set status="no" and leave answer empty
- If status="yes", the answer field must be non-empty
- If the evidence appears across adjacent sentences in the same retrieved document and the entity binding is explicit, you may answer from that document
- If the Sub-question asks "what country", "which country", "same country", or "country of origin", the answer must be only the country name. For text like "Quebec, Canada" answer "Canada"; for a clear demonym such as "Pakistani" answer "Pakistan".
- For country questions, do not answer with a continent, region, directional region, or cultural area such as "Eastern Europe", "Western Europe", "South Asia", or "the Middle East". If the evidence sentence frames the event with "In COUNTRY, ...", and the sub-question asks for the country where that event happens, answer that COUNTRY.
- If the Sub-question asks "what nationality", answer the nationality/demonym wording stated or directly implied by the document. Do not convert a nationality question to a country-name answer unless the document gives only the country name and no nationality adjective.
- If the Sub-question asks for the meaning/gloss of a word in a dictionary, answer the meaning/gloss, not the word form itself.
- If the Sub-question asks for a population or count and the best document only identifies the entity or location, status must be "no"; do not answer with that entity/location.
- For county/province/district containment questions, answer the requested administrative unit, not the broader state/country/region and not the contained town/district itself.
- For province-border questions, the answer must be a province. A country, lake, district, town, region, or the anchor province itself is the wrong type even if it borders the relevant district.
- For negative-relation questions such as "didn't discuss", "avoided", "refused to mention", or "did not talk about", answer the entity explicitly avoided/omitted. If a document says someone "avoided the topic of X", X is supported. Do not answer an entity that the same document says was discussed, stressed, promised, or focused on.
- For acronym or abbreviation questions asking what a name/topic/organization stands for, answer the expansion of the acronym/abbreviation. Do not answer an etymological name, native-language name, translation, or alternate name unless the sub-question explicitly asks for etymology or translation.

Retrieved Documents:
{docs_text}

Please respond in the following JSON format:
{{
  "status": "yes" or "no",
  "answer": "extracted answer if status is yes, empty if no",
  "analysis": "explain why you can/cannot answer based on the provided documents"
}}
Key principles:
- status="yes": Documents contain sufficient information
- status="no": Documents lack necessary information
- analysis: Always explain your reasoning
V35-style grounding notes:
- You have access to exactly {doc_count} documents, numbered [1] through [{doc_count}]
- Use only these retrieved documents; do not make up facts
- If the information is not in these documents, set status="no"
- Mark status="yes" when the retrieved documents directly answer the exact current sub-question, even if the original multi-hop question still needs later steps
- If documents provide related biography or context but not the requested entity, place, date, organization, number, or relation, set status="no"
- Do not substitute a different relation such as founded, created, directed, collaborated with, associated with, born in, or member of unless the sub-question itself asks for that relation
- Do not merge similarly named institutions, works, people, places, or organizations by reasoning that one is related to another. The document must explicitly establish the equivalence before you use its attributes.
- For founded/created/released/born/died/date questions, the date must be attached to the exact requested entity in the retrieved evidence. Dates attached only to a related entity are insufficient.
- For "which", "who", "what", and comparison questions, if the documents provide the entity/value needed to choose the answer, output the chosen answer rather than refusing because the wording is not identical
- For comparison sub-questions, answer only after the documents provide the compared values for all relevant entities. If one side's value is missing, status must be "no".
- For age comparisons, younger means the later birth date/year and older means the earlier birth date/year. If your reasoning says one date is later, the answer must be the entity with that later date for "younger".
- For death-date comparisons, died later means the later calendar death date/year and died first means the earlier calendar death date/year.
- For release-date comparisons, came out first/released first means the earlier release date/year, while came out later/released later means the later release date/year.
- When the sub-question asks for a country, return the country name if possible rather than a demonym/adjective
- If the documents only state a demonym or nationality adjective for a country question, use it as evidence and convert it to the country name when clear
- When the sub-question asks for nationality, preserve the nationality/demonym form when the document provides it; do not convert it to a country name.
- When the sub-question asks for an award, group, work, publication, or title, omit leading years and generic suffixes unless they are necessary to identify the answer
- When the sub-question asks for a source/publication/work that characterized, described, called, or named something, keep the source/publication/work title as the answer rather than replacing it with an inferred author.
- For alias questions, if a document says an entity is "also known as", "called", "nicknamed", or "referred to as" a phrase, return only that phrase as the answer. Do not return the entity's official/canonical name unless the sub-question asks for the official name.
- For quoted or exact title questions, prefer documents whose title exactly matches the requested title. Do not use a remake, adaptation, similarly named work, or title without/with a leading article when an exact title document is retrieved.
"""


def build_rewrite_prompt(
    original_question: str,
    subgoal: str,
    failure_info: str,
    documents: List[Document],
    previous_answers: Dict[int, str],
    *,
    max_docs: int,
    max_chars: int,
) -> str:
    docs_preview = format_documents(documents, max_docs=max_docs, max_chars=max_chars)
    previous_text = (
        "\n".join(f"Step {idx}: {answer}" for idx, answer in sorted(previous_answers.items()))
        if previous_answers
        else "None"
    )
    return f"""You are an expert query rewriter for information retrieval.

## Rewrite Task
Original Multi-hop Question: {original_question}
Current Sub-question: {subgoal}
Previous Sub-goal Answers:
{previous_text}
Failure Reason: {failure_info}

## Current Documents Preview
{docs_preview}

## Instructions
1. Analyze why the current query failed to retrieve relevant information
2. Generate an improved search query that changes the retrieval angle, not only synonyms
3. Focus on key entities, requested attributes, constraints, and previous answers
4. Keep the rewritten query concise but comprehensive
V35-style grounding notes:
- Include the key entities and the requested attribute from the question
- Do not introduce new entities or guess answers
- Preserve the original question's intent and constraints; do not add constraints that were not present
- Do not add current/currently/now unless the original question explicitly uses that constraint
- Keep the target of the Current Sub-question unchanged. If the Current Sub-question asks for an intermediate fact, rewrite a better query for that intermediate fact, not for the final answer.
- Never add the downstream/final relation from the Original Multi-hop Question to an intermediate sub-question query. If the Current Sub-question asks "Who is X's husband?" the rewritten query should search for X's husband, not for a law, date, place, award, or other later attribute of that husband.
- Do not jump to a later step. If the Current Sub-question asks "who is the spouse/husband/father/director/performer of X", the rewritten query must still search for that spouse/husband/father/director/performer, not for that person's birthplace, death date, award, or another downstream attribute.
- Prefer compact entity-attribute queries over verbose reasoning instructions
- If the previous query focused on a broad relation, use a different explicit attribute from the original question as an anchor
- If the current sub-question is ambiguous, include the original question's disambiguating constraint in the query
- If the current failure involves multiple candidates and a nationality, country, heritage, occupation, or other descriptor, include the candidate names plus that descriptor in the rewritten query
- If the current documents list multiple candidate answers for the relation, include those candidate names in the rewritten query together with the missing descriptor/attribute. Change the angle from the relation document to candidate verification, e.g. search candidate names plus nationality/occupation/date/place constraints rather than repeating only the original relation wording.
- If the retrieved documents identify the work type (album, song, film, book, etc.), preserve that work type in the rewrite; do not change an album into a song or a film into a book
- Do not return the same query unless it is already the shortest entity-attribute query possible

## Output JSON Format
{{
  "rewritten_query": "improved search query with expanded keywords",
  "strategy": "brief explanation of rewrite approach",
  "keywords": ["key", "terms", "and", "synonyms"]
}}

Generate rewrite:"""


def build_final_synthesis_prompt(question: str, step_answers: Dict[int, str], step_details: List[Dict[str, Any]] | None = None) -> str:
    if step_details:
        facts = "\n".join(
            "\n".join(
                [
                    f"Step {item.get('step_id')}:",
                    f"Sub-goal: {item.get('subgoal', '')}",
                    f"Resolved sub-goal: {item.get('filled_subgoal', '')}",
                    f"Answer: {item.get('answer', '')}",
                ]
            )
            for item in step_details
        )
    else:
        facts = "\n".join(f"Step {idx}: {answer}" for idx, answer in sorted(step_answers.items()))
    return f"""Original question: {question}

Executed sub-goals:
{facts}

Produce the final answer to the original question using only the executed sub-goal answers.
Use the chain of executed sub-goal answers to fill the placeholders and complete the original multi-hop reasoning.
Return the shortest answer phrase that directly answers the original question.
Do not include explanatory qualifiers, parenthetical context, or source wording in the answer field unless the qualifier is necessary to distinguish the entity.
Omit leading years or possessive date phrases for title/work/award answers unless the original question asks for the year.
If multiple executed answers are aliases for the same entity, prefer the full canonical name that includes the entity type requested by the original question, such as Airport, University, School, River, County, Province, City, Film, or Album. Do not shorten away the requested type word only to make the answer shorter.
If the original question asks "also known as", "known as", "another name", "called", "nicknamed", or "referred to as", return the alias/name phrase. This alias rule overrides the canonical-name preference.
For country questions, return the country name rather than a nationality adjective or demonym.
For nationality questions, return the nationality adjective or demonym when available rather than converting it to a country name.
Treat executed sub-goal answers as the available evidence; do not re-check retrieval from scratch.
Do not require additional proof beyond the executed sub-goal answers.
Choose the executed answer that matches the answer type requested by the original question.
For a linear chain where the final executed sub-goal already asks for the original requested attribute, return the final executed sub-goal answer.
If the final executed sub-goal is only a verification or supporting property check, return the earlier entity answer that the verification supports.
If the last executed sub-goal answer directly answers the original question after previous placeholders are resolved, return that last answer.
If the sub-goal needed to answer the original question failed or was not executed, do not answer with an intermediate anchor entity. Return "Not found in retrieved documents" unless another completed sub-goal directly answers the original requested relation and answer type.
If the original question asks for a county, province, district, capital, airport, university, or other typed entity, the answer field must be that typed entity. Do not return a broader state/country/region or a shorter alias that omits the requested type when a completed answer contains the typed name.
For counterpart questions using "shares a border with", "borders", "with", or "between", return the other entity of the requested type. Never return the anchor entity itself as the final answer.
For "who is part/member of the group..." questions, return the person/member resolved by the member sub-goal, not the group that performed or created the work.
Before returning JSON, check that the answer field is the same entity/value your analysis identifies as the answer. If your analysis says the answer is a county/province/capital/person/work, put that exact item in the answer field.
Return "yes" or "no" only when the original question is explicitly a yes/no question or asks whether something is true.
For "same" property yes/no questions, return "yes" when the executed property answers are equivalent, and "no" when they differ.
If the executed property answers are the same normalized phrase, that is sufficient for "yes".
Never answer "no" for a same-property question when all executed property answers are identical.
For same-country questions, compare the country component only, not the state, province, city, airport, or local place. Different states/provinces/cities inside the same country still mean the answer is "yes".
If your analysis says the entities are in the same country, the answer must be "yes"; if it says they are in different countries, the answer must be "no".
For "which" comparison questions, return the name of the item/entity that satisfies the comparison, never "yes" or "no".
For "Which [entity type] whose [related attribute] is comparative, A or B?" questions, return the candidate entity A or B, not the related attribute value such as the director, author, date, country, or number used to compare them.
If the final executed sub-goal asks for a word's meaning/gloss, return that final meaning/gloss answer. Do not replace it with an earlier intermediate religion, people, language, or source word.
If the final executed sub-goal gives an Arabic equivalent but the original asks for the meaning in a dictionary, return the meaning/concept that the Arabic expression refers to, not the Arabic expression itself.
For characterization/description/naming questions, return the explicitly resolved agent type from the executed sub-goal. If the executed answer is a publication/source/work title, do not convert it to an inferred author.
For comparison questions:
- younger means the later birth year/date
- older means the earlier birth year/date
- born first means the earlier birth year/date
- born later means the later birth year/date
- died later, death later, came later, released later, and later date mean the later calendar date/year
- died first, death first, came first, released first, and earlier date mean the earlier calendar date/year
- If both release years and birth years/dates are present, use the birth years/dates when the original question compares people by age or birth order
Before returning a comparison or yes/no answer, check that the answer is consistent with your own analysis.
Only say "Not found in retrieved documents" when the executed answers genuinely lack the information needed for the original question. Do not say it when the last executed answer is a concrete non-empty answer to the requested attribute.
Put explanation only in analysis.

Return JSON only:
{{
  "answer": "final answer",
  "analysis": "brief reasoning based on the executed sub-goals"
}}"""
