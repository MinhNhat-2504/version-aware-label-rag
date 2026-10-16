"""Assistant-drafted expansion questions, written before any model run on these labels.

One question per frozen label pair is applied unchanged to both pinned versions. The
drafts were written from the frozen section text only; no retrieval or answering model
has seen these labels. Two independent readers must accept, rewrite or reject each one.
"""

QUESTIONS: dict[str, str] = {
    # Consecutive arm: sampled regardless of content change.
    "04a12e4b-32ba-4cf5-9451-23340e83f759": "How is this methylprednisolone 4 mg tablet product packaged, and which NDC is listed for that package?",
    "05fd0c36-39b4-4a43-8490-eb8dfc0920ae": "What is the recommended dosing frequency for tobramycin and dexamethasone ophthalmic suspension, and how may it be changed during the first 24 to 48 hours?",
    "1090762c-eec1-4ab0-9ddc-43caa764cd0e": "What is the recommended initial dose of calcium acetate for adult dialysis patients, and how should the dose then be adjusted?",
    "27b0c628-820a-48bf-a411-86c0ebd4cc8d": "Which capsule strengths of amlodipine and benazepril hydrochloride does this label describe?",
    "28f79c75-82d2-4912-81fc-a723b5ccf4c1": "What preparation instructions does this Sodium Chloride Injection label give to reduce the risk of air embolism?",
    "2c7b70d0-fdd1-00c7-e063-6394a90a637e": "Who may administer nitrous oxide according to this label, and which fire-related hazard does the label state?",
    "2c7b9b54-d778-579d-e063-6294a90aefaf": "Who may administer Medical Air according to this label, and what does the label warn about combustion?",
    "30e78ac6-64e3-41d0-8097-cb250c7b3739": "What breathing hazard does this compressed nitrogen label warn about, and where should the gas be used and stored?",
    "3173e592-6f28-4864-9bf0-0ef0452da3de": "What washout period does this sacubitril and valsartan label require when switching from an ACE inhibitor?",
    "31b347d2-f156-4055-9d8f-7cf0df420296": "What is EMFLAZA indicated for, and what is its recommended oral dosage?",
    "34c02bcc-d6ff-99d4-8e6a-4a91908adca8": "What is this lamivudine tablet product indicated for, and what limitation of use does the label state about hepatitis B?",
    "37f04f8b-001e-66ff-4574-cc34cebce61c": "What does this teriflunomide label warn about liver injury and about use in pregnancy?",
    "4a75ebac-a765-460d-bb6a-9fbabb0b927d": "For how long is Miudella indicated to prevent pregnancy, and by when must it be removed after insertion?",
    "4d0bfd28-460a-48ba-bd17-cf75e56d62f4": "What is this Sodium Chloride Injection indicated for, including any use in hemodialysis procedures?",
    "4d682f98-1e8a-4c2b-6df3-169d2ed10dc1": "What does this etodolac label say about the combined effect of warfarin and NSAIDs on gastrointestinal bleeding?",
    "4ecf9314-76e2-c270-e063-6294a90ad8db": "Which bottle counts are listed for this cephalexin capsule product?",
    "5127517f-b8d0-4af1-8a2f-5225719600bc": "What is the recommended starting dose of mirtazapine orally disintegrating tablets, and when should it preferably be taken?",
    "5f6c3eed-2a5d-46ed-9a7c-59715f338a15": "What does this nevirapine label say about a pregnancy exposure registry and how to enroll patients?",
    "6584d696-7d32-4744-b225-03c52803e16a": "Under what circumstances does this oxygen label say the product is for emergency use only?",
    "68024459-987a-9cf4-f539-4959093bc9f0": "What are fenofibrate capsules indicated for, and what limitation of use does the label give about markedly elevated triglycerides?",
    "7293cef3-ef9e-40dc-9995-875ef047f693": "How should the prazosin dose be adjusted when a diuretic or another antihypertensive agent is added?",
    "753453c5-8a7a-878e-3ce7-1d1edce718a9": "What is the recommended daily dosage of leflunomide, and what determines whether treatment starts with a loading dose?",
    "76e1833c-e105-5a3d-0f01-1330d8cc7ae1": "What does the pregnancy risk summary of this dasatinib label say about human data and fetal adverse effects?",
    "78f39126-3e3b-4633-a4e0-c1f09a794eec": "What is the recommended dose of dorzolamide hydrochloride and timolol maleate ophthalmic solution?",
    "7c22a2d6-821c-4f83-aead-e756aed930e5": "Which conditions are these naproxen sodium controlled-release tablets indicated to treat?",
    "7de9b8e0-195a-b59e-7bbf-3e95c4096af9": "Is this sacubitril and valsartan product indicated for pediatric heart failure, and for which ages?",
    "8223c44a-73a9-4da5-a468-d1d8bdb0bbd6": "What is hydralazine hydrochloride injection indicated for, and what is the usual dose?",
    "87095352-c088-74b1-7be0-d3e293a90611": "What are lamivudine tablets (HBV) indicated for, and what does the label say about HIV-1 resistance in patients with unrecognized HIV-1 infection?",
    "89a5ec53-d956-4329-8004-0f40f51c88a3": "For which pediatric patients is ISENTRESS indicated, including any minimum body weight?",
    "8b8fe00e-7a2f-486b-9b72-465a494d8939": "How long must elapse between stopping an MAOI and starting this fluoxetine product, and between stopping fluoxetine and starting an MAOI?",
    "8c8bcba9-eaeb-aa44-f9ea-b580de55a439": "Which inactive ingredients are listed for these sertraline hydrochloride capsules?",
    "94d941e3-d3d8-4414-bd5c-4a92848e738f": "What does this minocycline label say about use during tooth development?",
    # Text-change-enriched arm: a clinical-heading section changed between versions.
    "9502860d-0261-4d69-a4be-827a5376d356": "What does this octreotide acetate injection label say about steatorrhea and malabsorption of dietary fats, and what should patients be advised to do?",
    "a4d89cd9-c5d8-490f-996d-fc15b47c7b21": "Besides age, which factors does this lamotrigine label identify as possibly increasing the risk or severity of rash?",
    "aa06e286-75c6-4d7a-959f-6008d38aa6e5": "What does this Sodium Chloride Injection label instruct to reduce the risk of air embolism, including anything about pressure infusion?",
    "b3bb8222-161e-4d9c-8e08-7e22ad65bf72": "What is the recommended cephalexin capsule dosage for adults and pediatric patients at least 15 years of age?",
    "b6b8a10c-64c3-4856-8961-d72d68784f5d": "How long before surgery should dapagliflozin be withheld, and what should a patient do after a missed dose?",
    "eb811d79-fdb8-1f1a-c2ad-1cc4bfc7eadf": "Which carton pack sizes are listed for the 250 mL container of this Sodium Chloride Injection?",
    "f550d2ba-49ac-4a3b-95db-692b468af479": "What does this dexmedetomidine label say about diabetes insipidus during sedation, including its signs?",
    "a363f0cd-9c8e-4622-9271-171706846b96": "What does this methylprednisolone acetate label say about concomitant use of amphotericin B and hydrocortisone?",
    "a88cd5b5-bdcb-4ef7-8b04-ee4603bcd621": "What does this midodrine label state about pregnancy, including any pregnancy category and animal findings?",
    "b509d0eb-fe01-4902-89ae-a0dc7f7c7f93": "What limitation of use does this amoxicillin and clavulanate potassium label state when susceptibility to amoxicillin alone is shown?",
    "5409c8ed-17c3-454f-bb88-d597ba8f84ec": "What does this abacavir label say about a pregnancy exposure registry and how healthcare providers can register patients?",
    "13e15ee0-d679-4fa9-9430-e2e2170474da": "What dosage modifications for adverse reactions does this WELIREG label give when WELIREG is used in combination with lenvatinib?",
    "170ec388-d399-4543-8743-208394b38904": "What does the pregnancy subsection of this estradiol tablet label say about use during pregnancy?",
    "23126370-8d6a-4027-bd8f-860e2776ba3a": "Which antibacterials does this itraconazole label list as increasing itraconazole concentrations, and what management does it recommend?",
    "2506895d-f3e0-4be9-95cb-7a6fd44de46d": "What does the pregnancy subsection of this cephalexin label say about available human data on drug-associated risks?",
    "2f5f06de-ee98-44c0-be7e-9a998d1e5c16": "What does this SUPREP Bowel Prep Kit label say about hypersensitivity, including any contraindication or warning about hypersensitivity reactions?",
}
