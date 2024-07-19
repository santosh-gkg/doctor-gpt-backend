import os
from fastapi import FastAPI, Request

from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from langchain_groq import ChatGroq

from dotenv import load_dotenv
from langchain_community.embeddings import GPT4AllEmbeddings
# from openai import OpenAI 
from groq import Groq
# from langchain_groq import ChatGroq
from langchain.vectorstores import Chroma


load_dotenv()


###### CORS ######
app = FastAPI()
origins = [
    "http://localhost:3000",  # React app origin
    "http://localhost:5173",  # Vite default dev server port
    "http://sgkg.tech"
    "http://sgkg.tech/"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

######## GPT4All Embeddings ########

model_name = "all-MiniLM-L6-v2.gguf2.f16.gguf"
gpt4all_kwargs = {'allow_download': 'True'}
embeddings = GPT4AllEmbeddings(
    model_name=model_name,
    gpt4all_kwargs=gpt4all_kwargs
)


######## Chroma Vectorstore ########

vectordb = Chroma(persist_directory='larger', embedding_function=embeddings)

retriever = vectordb.as_retriever()

#### PROMPT GENERATOR ####

def prompt_generator(prompt):
    context_list=retriever.invoke(prompt)
    context=""
    for i in range(0,3):
        context+=context_list[i].page_content+"\n"
        
    statement=f"Here is the context for your query: <context> \n {context} \n <context>"
    statement+=" using  this context answer the question below "
    statement+=f"<question> : \n {prompt} \n <question>"
    statement+=" \n Please provide the answer to the question above"
    return statement,context


######## AI Client  ########
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)


######## API Endpoints ########
class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[Message]



### few shots ####

few_shots=[
    {'role':'user',
    'content':'''Here is the context for your query: <context> \n before age of 50 years due to heart disease in one or more\nrelatives\n7. Disability from heart disease in a close relative before age\nof 50 years\n8. Specific knowledge of certain cardiac conditions in family\nmembers: hypertrophic cardiomyopathy, dilated cardiomy-\nopathy, long QT syndrome or other ion channelopathies,\nMarfan syndrome, or other important arrhythmias\nphysical Examination\n9. Heart murmur\n10. Diminished femoral pulse (to exclude coarctation)\n11. Phenotype of Marfan syndrome\n12. Brachial artery blood pressure (sitting position)\nReproduced with permission from Lawless CE, Asplund C, Asif IM,\net al. Protecting the heart of the American athlete: proceedings of\nthe American College of Cardiology Sports and Exercise Cardiology\nThink Tank October 18, 2012, Washington, DC. J Am Coll Cardiol.\n2014;64(20):2146–2171.CMDT23_Ch10_p0323-p0441.indd \n439 30/06/22 3:10 PM\nHistory of ischemic heart disease\nHistory of heart failure\nInsulin treatment for diabetes mellitus\nSerum creatinine level > 2 mg/dL (> 176.8 mcmol/L)\nHistory of cerebrovascular disease\nScoring (Number of\nPredictors Present)Risk of Major Cardiac\nComplications1None\n0.4%\nOne 1%\nTwo 2.4%\nMore than two 5.4%1\nCardiac death, MI, or nonfatal cardiac arrest.\nData from Devereaux PJ et al. Perioperative cardiac events in\npatients undergoing noncardiac surgery: a review of the magni-\ntude of the problem, the pathophysiology of the events and\nmethods to estimate and communicate risk. CMAJ. 2005;173:627.CMDT23_Ch03_p0042-p0051.indd \n43 25/06/22 2:41 PM\nCHApTER 10440 CMDT 2023T\nable 10–20. Recommendations or competitive sports participation among athletes with potential causes o SCD.Condition\n36th Bethesda Conference Euroean Society of Cardiology\nStructural Cardiac Abnormalities\nHCM Exclude athletes with a probable or definitive\nclinical diagnosis from all competitive sports.\nGenotype-positive/phenotype-negative\nathletes may still compete.Exclude athletes with a probable or definitive clinical\ndiagnosis from all competitive sports.\nExclude genotype-positive/phenotype-negative\nindividuals from competitive sports.\nARVC Exclude athletes with a probable or definitive\ndiagnosis from competitive sports.Exclude athletes with a probable or definitive\ndiagnosis from competitive sports.\nCCAA Exclude from competitive sports. Not applicable.\nParticipation in all sports 3 months after successful\nsurgery would be permitted for an athlete\nwith ischemia, ventricular arrhythmia or\ntachyarrhythmia, or LV dysfunction during maximal\nexercise testing.\nElectrical Cardiac Abnormalities\nWPW Athletes without structural heart disease, without\na history of palpitations, or without tachycardia\ncan participate in all competitive sports.\nIn athletes with symptoms, electrophysiological\nstudy and ablation are recommended. Return to\ncompetitive sports is allowed after corrective ablation,\nprovided that the ECG has normalized.Athletes without structural heart disease,\nwithout a history of palpitations, or without\ntachycardia can participate in all competitive\nsports.\nIn athletes with symptoms, electrophysiological\nstudy and ablation are recommended. Return to\ncompetitive sports is allowed after corrective\nablation, provided that the ECG has normalized.\nLQTS Exclude any athlete with a previous cardiac arrest\nor syncopal episode from competitive sports.\nAsymptomatic patients restricted to competitive\nlow-intensity sports.\nGenotype-positive/phenotype-negative athletes\nmay still compete.Exclude any athlete with a clinical or genotype\ndiagnosis from competitive sports.\nBrS Exclude from all competitive sports except those\nof low intensity.Exclude from all competitive sports.\nCPVT Exclude all patients with a clinical diagnosis from\ncompetitive sports.\nGenotype-positive/phenotype-negative patients\nmay still compete in low-intensity sports.Exclude all patients with a clinical diagnosis\nfrom competitive sports.\nGenotype-positive/phenotype-negative\npatients are also excluded.\nAcquired Cardiac Abnormalities\nCommotio cordis Eligibility for returning to competitive sport in survivors\nis a matter of individual clinical judgment. Survivors\nmust undergo a thorough cardiovascular workup\nincluding 12-lead ECG, ambulatory ECG monitoring,\nand echocardiography.Not applicable.\nMyocarditis Exclude from all competitive sports.\nConvalescent period of 6 months.\nAthletes may return to competition when test\nresults normalize.Exclude from all competitive sports.\nConvalescent period of 6 months.\nAthletes may return to competition when\ntest results normalize.\nARVC, arrhythmogenic right ventricular cardiomyopathy; BrS, Brugada syndrome; CCAA, congenital coronary artery anomalies; CPVT, cat-\necholaminergic polymorphic ventricular tachycardia; HCM, hypertrophic cardiomyopathy; LQTS, long QT syndrome; SCD, sudden cardi ac\ndeath; WPW, Wolff-Parkinson-White syndrome.\nReproduced with permission from Chandra N et al. Sudden cardiac death in young athletes: practical challenges and diagnostic di lemmas.\nJ Am Coll Cardiol. 2013;61(10):1027–1040.CMDT23_Ch10_p0323-p0441.indd \n440 30/06/22 3:10 PM\n \n <context> using  this context answer the question below <question> : \n heart failure \n <question> \n Please provide the answer to the question above
    
    '''},
    {'role':'assistant',
    'content':''' #Heart Failure \n\nHeart failure is a condition where the heart cannot pump blood efficiently to meet the body's needs. It can be caused by various factors, including high blood pressure, coronary artery disease, heart valve problems, cardiomyopathy, and congenital heart defects.\n\nIn the context you provided, there are several risk factors for major cardiac complications, one of which is a history of heart failure. The presence of multiple risk factors increases the likelihood of cardiac complications during noncardiac surgery.\n\nRegarding sports participation, athletes with a history of heart failure are generally advised to avoid competitive sports, as they have an increased risk of cardiac events. However, individual clinical judgment is essential, and athletes may return to competition when test results normalize after a convalescent period, such as in myocarditis.\n\nIt is important to consult a healthcare professional for personalized advice and recommendations regarding heart failure and sports participation.
'''
    }
]

@app.post("/doctor/")
async def chat(request: ChatRequest):
    messages = request.messages
    user_message = messages[-1].content
    # return {"response": messages}

    newprompt,context=prompt_generator(user_message)
    stream = client.chat.completions.create(
        model="mixtral-8x7b-32768",
        messages=[
            {"role": "system", "content": "You are an advanced AI doctor chatbot designed to assist users with medical inquiries, provide health-related advice, and offer support in understanding medical conditions. Your role is to deliver accurate, empathetic, and actionable information based on the user's input while adhering to medical guidelines and ethical standards. Your responses should always be fact-checked, up-to-date, and compliant with general medical practices."},
        ] +few_shots+
        [
            {"role": m.role, "content": m.content}
            for m in messages[:-1]
        ] + [{"role": "user", "content": newprompt}],
        temperature=0.0,
    )

    response = stream.choices[0].message.content
    return {"response": response,'context':newprompt}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)


