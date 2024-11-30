from enum import Enum
import os
from typing import List

import numpy as np
from dotenv import load_dotenv

from pydantic import BaseModel

import utils

class LogisticRegression:
    """
    A model class for logistic regression. To accomodate the multilabeled data, 
    sklearns LogisticRegression class is wrapped in an sklearn MultiOutputClassifier.

    functions:
    - fit: fit takes the train variables x_train and y_train and fits the logistic regression to the training data.
    - predict: makes predictions on the x_test set.

    attributes:
    - model: model stores the sklearn model object.
    """
    def __init__(self, solver= "liblinear", class_weight="balanced", max_iter=1000):
        from sklearn.linear_model import LogisticRegression
        from sklearn.multioutput import MultiOutputClassifier
        self.model = MultiOutputClassifier(LogisticRegression(solver=solver,class_weight=class_weight,max_iter=max_iter))
    
    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
    
    def predict(self, X_test):
        return self.model.predict(X_test)

class SVM:
    ...
 
class GPT:
    # https://gist.github.com/daveebbelaar/d65f30bd539a9979d9976af80ec41f07 
    def __init__(self, system_prompt, model:str = "gpt-4o-mini"):
        from openai import OpenAI
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.system_prompt = system_prompt
        self.model = model

    # fit function such that the pipe line does not break. 
    # The GPT models aren't trained, so this is skipped.
    def fit(self, X, y):
        pass 

    def predict(self, X):
        texts = X
        responses = []
        sub_mlb = utils.load_sub_mlb()

        for text in texts:
            response = self.client.beta.chat.completions.parse(
                model=self.model,
                response_format=ResponseModel,
                messages=[
                    {"role":"system", "content": self.system_prompt},
                    {"role":"user","content": text}
                ]
            )
            
            narrs = [narr.value for narr in response.choices[0].message.parsed.narratives]
            response = sub_mlb.transform([narrs])
            responses.append(response[0])
        
        return responses

class GPT_ensemble:
    def __init__(self, system_prompts:list[str], model_names:list[str], aggregation_method:str = "max"):
        assert len(system_prompts) == len(model_names)

        self.aggregation_method = aggregation_method
        self.prompts = system_prompts
        self.model_names = model_names
        self.GPTs = []

        for i in range(len(self.prompts)):
            self.GPTs.append(GPT(self.prompts[i], self.model_names[i]))

    def fit(self, X, y):
        pass

    def predict(self, X_test):
        model_outputs = []
        for GPT in self.GPTs:
            model_outputs.append(GPT.predict(X_test))

        # do some aggregation or avging or sumthing
        if self.aggregation_method == "max":
            model_outputs = np.any(model_outputs, axis = 0)

        #sub_mlb = utils.load_sub_mlb()
        #model_outputs = sub_mlb.transform(model_outputs)

        return model_outputs


# classes that define a reponse format for the GPT model, to save time, I generated the enum for these with gpt-4o
class SubNarratives(str,Enum):
    OTHER = "Other"

    # Amplifying Climate Fears
    CC_AMPLIFY_FEAR_EXISTING = "CC: Amplifying Climate Fears: Amplifying existing fears of global warming"
    CC_AMPLIFY_FEAR_DOOMSDAY = "CC: Amplifying Climate Fears: Doomsday scenarios for humans"
    CC_AMPLIFY_FEAR_UNINHABITABLE = "CC: Amplifying Climate Fears: Earth will be uninhabitable soon"
    CC_AMPLIFY_FEAR_OTHER = "CC: Amplifying Climate Fears: Other"
    CC_AMPLIFY_FEAR_TOO_LATE = "CC: Amplifying Climate Fears: Whatever we do it is already too late"
    
    # Climate Change is Beneficial
    CC_BENEFICIAL_CO2 = "CC: Climate change is beneficial: CO2 is beneficial"

    # Controversy About Green Technologies
    CC_GREEN_TECH_COSTLY = "CC: Controversy about green technologies: Renewable energy is costly"
    CC_GREEN_TECH_DANGEROUS = "CC: Controversy about green technologies: Renewable energy is dangerous"
    CC_GREEN_TECH_UNRELIABLE = "CC: Controversy about green technologies: Renewable energy is unreliable"
    CC_GREEN_TECH_OTHER = "CC: Controversy about green technologies: Other"

    # Criticism of Climate Movement
    CC_CRITICISM_MOVEMENT_ACTIVISTS = "CC: Criticism of climate movement: Ad hominem attacks on key activists"
    CC_CRITICISM_MOVEMENT_ALARMIST = "CC: Criticism of climate movement: Climate movement is alarmist"
    CC_CRITICISM_MOVEMENT_CORRUPT = "CC: Criticism of climate movement: Climate movement is corrupt"
    CC_CRITICISM_MOVEMENT_OTHER = "CC: Criticism of climate movement: Other"

    # Criticism of Climate Policies
    CC_CRITICISM_POLICIES_INEFFECTIVE = "CC: Criticism of climate policies: Climate policies are ineffective"
    CC_CRITICISM_POLICIES_PROFIT = "CC: Criticism of climate policies: Climate policies are only for profit"
    CC_CRITICISM_POLICIES_ECONOMIC_IMPACT = "CC: Criticism of climate policies: Climate policies have negative impact on the economy"
    CC_CRITICISM_POLICIES_OTHER = "CC: Criticism of climate policies: Other"

    # Criticism of Institutions and Authorities
    CC_CRITICISM_AUTHORITIES_INTERNATIONAL = "CC: Criticism of institutions and authorities: Criticism of international entities"
    CC_CRITICISM_AUTHORITIES_GOVERNMENTS = "CC: Criticism of institutions and authorities: Criticism of national governments"
    CC_CRITICISM_AUTHORITIES_POLITICAL = "CC: Criticism of institutions and authorities: Criticism of political organizations and figures"
    CC_CRITICISM_AUTHORITIES_EU = "CC: Criticism of institutions and authorities: Criticism of the EU"
    CC_CRITICISM_AUTHORITIES_OTHER = "CC: Criticism of institutions and authorities: Other"

    # Downplaying Climate Change
    CC_DOWNPLAY_CO2_CONCENTRATIONS = "CC: Downplaying climate change: CO2 concentrations are too small to have an impact"
    CC_DOWNPLAY_NATURAL_CYCLES = "CC: Downplaying climate change: Climate cycles are natural"
    CC_DOWNPLAY_HUMAN_IMPACT = "CC: Downplaying climate change: Human activities do not impact climate change"
    CC_DOWNPLAY_ADAPTATION = "CC: Downplaying climate change: Humans and nature will adapt to the changes"
    CC_DOWNPLAY_ICE_NOT_MELTING = "CC: Downplaying climate change: Ice is not melting"
    CC_DOWNPLAY_OTHER = "CC: Downplaying climate change: Other"
    CC_DOWNPLAY_TEMP_IMPACT = "CC: Downplaying climate change: Temperature increase does not have significant impact"
    CC_DOWNPLAY_COOLING_TREND = "CC: Downplaying climate change: Weather suggests the trend is global cooling"

    # Green Policies as Geopolitical Instruments
    CC_GEO_POLICIES_NEO_COLONIALISM = "CC: Green policies are geopolitical instruments: Green activities are a form of neo-colonialism"
    CC_GEO_POLICIES_OTHER = "CC: Green policies are geopolitical instruments: Other"

    # Hidden Plots by Secret Schemes of Powerful Groups
    CC_PLOTS_GLOBAL_ELITES = "CC: Hidden plots by secret schemes of powerful groups: Blaming global elites"
    CC_PLOTS_HIDDEN_MOTIVES = "CC: Hidden plots by secret schemes of powerful groups: Climate agenda has hidden motives"
    CC_PLOTS_OTHER = "CC: Hidden plots by secret schemes of powerful groups: Other"

    # Questioning the Measurements and Science
    CC_QUESTION_DATA_NO_INCREASE = "CC: Questioning the measurements and science: Data shows no temperature increase"
    CC_QUESTION_GREENHOUSE_EFFECT = "CC: Questioning the measurements and science: Greenhouse effect/carbon dioxide do not drive climate change"
    CC_QUESTION_METHODS_UNRELIABLE = "CC: Questioning the measurements and science: Methodologies/metrics used are unreliable/faulty"
    CC_QUESTION_OTHER = "CC: Questioning the measurements and science: Other"
    CC_QUESTION_COMMUNITY_UNRELIABLE = "CC: Questioning the measurements and science: Scientific community is unreliable"

    # Amplifying War-Related Fears
    WWIII = "URW: Amplifying war-related fears: By continuing the war we risk WWIII"
    NATO_INTERVENE = "URW: Amplifying war-related fears: NATO should/will directly intervene"
    WAR_FEARS_OTHER = "URW: Amplifying war-related fears: Other"
    RUSSIA_ATTACK_COUNTRIES = "URW: Amplifying war-related fears: Russia will also attack other countries"
    NUKES = "URW: Amplifying war-related fears: There is a real possibility that nuclear weapons will be employed"
    
    # Blaming the War on Others Rather than the Invader
    NOT_INVADER_OTHER = "URW: Blaming the war on others rather than the invader: Other"
    WEST_AGGRESSORS = "URW: Blaming the war on others rather than the invader: The West are the aggressors"
    UKRAINE_AGGRESSORS = "URW: Blaming the war on others rather than the invader: Ukraine is the aggressor"
    
    # Discrediting Ukraine
    DISCREDIT_UA_GOVERNMENT = "URW: Discrediting Ukraine: Discrediting Ukrainian government and officials and policies"
    DISCREDIT_UA_MILITARY = "URW: Discrediting Ukraine: Discrediting Ukrainian military"
    DISCREDIT_UA_SOCIETY = "URW: Discrediting Ukraine: Discrediting Ukrainian nation and society"
    DISCREDIT_UA_OTHER = "URW: Discrediting Ukraine: Other"
    DISCREDIT_UA_HISTORY = "URW: Discrediting Ukraine: Rewriting Ukraine’s history"
    DISCREDIT_UA_HOPELESS = "URW: Discrediting Ukraine: Situation in Ukraine is hopeless"
    DISCREDIT_UA_CRIMINAL = "URW: Discrediting Ukraine: Ukraine is a hub for criminal activities"
    DISCREDIT_UA_PUPPET = "URW: Discrediting Ukraine: Ukraine is a puppet of the West"
    DISCREDIT_UA_NAZISM = "URW: Discrediting Ukraine: Ukraine is associated with nazism"
    
    # Discrediting the West, Diplomacy
    DISCREDIT_WEST_DIPLOMACY_FAIL = "URW: Discrediting the West, Diplomacy: Diplomacy does/will not work"
    DISCREDIT_WEST_DIPLOMACY_OTHER = "URW: Discrediting the West, Diplomacy: Other"
    DISCREDIT_WEST_DIVIDED = "URW: Discrediting the West, Diplomacy: The EU is divided"
    DISCREDIT_WEST_INTERESTS = "URW: Discrediting the West, Diplomacy: The West does not care about Ukraine, only about its interests"
    DISCREDIT_WEST_OVERREACTING = "URW: Discrediting the West, Diplomacy: The West is overreacting"
    DISCREDIT_WEST_WEAK = "URW: Discrediting the West, Diplomacy: The West is weak"
    DISCREDIT_WEST_TIRED = "URW: Discrediting the West, Diplomacy: West is tired of Ukraine"
    
    # Distrust Towards Media
    DISTRUST_MEDIA_OTHER = "URW: Distrust towards Media: Other"
    DISTRUST_MEDIA_UA = "URW: Distrust towards Media: Ukrainian media cannot be trusted"
    DISTRUST_MEDIA_WEST = "URW: Distrust towards Media: Western media is an instrument of propaganda"
    
    # Hidden Plots by Secret Schemes of Powerful Groups
    PLOTS_OTHER = "URW: Hidden plots by secret schemes of powerful groups: Other"
    
    # Negative Consequences for the West
    NEGATIVE_WEST_OTHER = "URW: Negative Consequences for the West: Other"
    NEGATIVE_WEST_SANCTIONS_BACKFIRE = "URW: Negative Consequences for the West: Sanctions imposed by Western countries will backfire"
    NEGATIVE_WEST_REFUGEES = "URW: Negative Consequences for the West: The conflict will increase the Ukrainian refugee flows to Europe"
    
    # Overpraising the West
    OVERPRAISING_WEST_OTHER = "URW: Overpraising the West: Other"
    OVERPRAISING_WEST_HISTORY = "URW: Overpraising the West: The West belongs in the right side of history"
    OVERPRAISING_WEST_SUPPORT = "URW: Overpraising the West: The West has the strongest international support"
    
    # Praise of Russia
    PRAISE_RUSSIA_OTHER = "URW: Praise of Russia: Other"
    PRAISE_RUSSIA_PUTIN = "URW: Praise of Russia: Praise of Russian President Vladimir Putin"
    PRAISE_RUSSIA_MILITARY = "URW: Praise of Russia: Praise of Russian military might"
    PRAISE_RUSSIA_SUPPORT = "URW: Praise of Russia: Russia has international support from a number of countries and people"
    PRAISE_RUSSIA_PEACE = "URW: Praise of Russia: Russia is a guarantor of peace and prosperity"
    PRAISE_RUSSIA_INVASION_SUPPORT = "URW: Praise of Russia: Russian invasion has strong national support"
    
    # Russia is the Victim
    RUSSIA_VICTIM_OTHER = "URW: Russia is the Victim: Other"
    RUSSIA_SELF_DEFENSE = "URW: Russia is the Victim: Russia actions in Ukraine are only self-defence"
    RUSSIA_RUSSO_PHOBIC = "URW: Russia is the Victim: The West is russophobic"
    RUSSIA_ANTI_RU = "URW: Russia is the Victim: UA is anti-RU extremists"
    
    # Speculating War Outcomes
    WAR_OUTCOMES_OTHER = "URW: Speculating war outcomes: Other"
    WAR_OUTCOMES_RU_COLLAPSING = "URW: Speculating war outcomes: Russian army is collapsing"
    WAR_OUTCOMES_UA_COLLAPSING = "URW: Speculating war outcomes: Ukrainian army is collapsing"

class ResponseModel(BaseModel):
    narratives: List[SubNarratives]
    explanation: str 