from allennlp import *
from allennlp.predictors.predictor import Predictor
predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/bidaf-model-2020.03.19.tar.gz")

#textual data 
passage = """
Captain America is a fictional superhero appearing in American comic books published by Marvel Comics. Created by cartoonists Joe Simon and Jack Kirby, 
the character first appeared in Captain America Comics #1 (cover dated March 1941) from Timely Comics, a predecessor of Marvel Comics. 
Captain America was designed as a patriotic supersoldier who often fought the Axis powers of World War II and was Timely Comics' most popular character during the wartime period. 
The popularity of superheroes waned following the war, and the Captain America comic book was discontinued in 1950, with a short-lived revival in 1953. 
Since Marvel Comics revived the character in 1964, Captain America has remained in publication.
The character wears a costume bearing an American flag motif, and he utilizes a nearly indestructible shield that he throws as a projectile. 
Captain America is the alter ego of Steve Rogers, a frail young man enhanced to the peak of human perfection by an experimental serum to aid the United States government's efforts 
in World War II. Near the end of the war, he was trapped in ice and survived in suspended animation until he was revived in modern times. Although Captain America often struggles
to maintain his ideals as a man out of his time, he remains a highly respected figure in his community, which includes becoming the long-time leader of the Avengers. 
Captain America was the first Marvel Comics character to appear in media outside comics with the release of the 1944 movie serial, 
Captain America. Since then, the character has been featured in other films and television series. In the Marvel Cinematic Universe (MCU), the character is portrayed by Chris Evans.
"""


result=predictor.predict(
  passage=passage,
  question="who created this comic?"  #you can enter the questions here to see how the framwork procure results and the accuracy of answers
)
print(result['best_span_str'])
