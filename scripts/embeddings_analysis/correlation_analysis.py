MORAL_TRAIT_SET = [11, 14, 25, 28, 31, 38, 39, 79, 81, 84, 101, 118, 121, 154, 195, 222, 227, 304, 362, 396, 434, 446, 448, 450, 489, 494]
MORAL_TRAIT_COLS = [f"trait_{i}" for i in MORAL_TRAIT_SET]

# These are the traits corresponding to the indices in MORAL_TRAIT_SET
# NOTE: We have more but these are the ones that have moral valence.
trait_dict = {
    11: "innocent-worldly",
    14: "cunning-honorable",
    25: "forgiving-vengeful",
    28: "loyal-traitorous",
    31: "rude-respectful",
    38: "arrogant-humble",
    39: "heroic-villainous",
    79: "selfish-altruistic",
    81: "angelic-demonic",
    84: "cruel-kind",
    101: "biased-impartial",
    118: "jealous-compersive",
    121: "sarcastic-genuine",
    154: "judgemental-accepting",
    195: "complimentary-insulting",
    222: "wholesome-salacious",
    227: "racist-egalitarian",
    304 : "narcissistic-low self esteem",
    362: "jealous-opinionated",
    396: "innocent-jaded",
    434: "resentful-euphoric",
    446: "inappropriate-seemly",
    448: "fake-real",
    450: "catty-supportive",
    489: "sincere-irreverent",
    494: "hopeful-fearful"
}

full_default_trait_dict = {
  "1": "playful-serious",
  "2": "shy-bold",
  "3": "cheery-sorrowful",
  "4": "masculine-feminine",
  "5": "charming-awkward",
  "6": "lewd-tasteful",
  "7": "intellectual-physical",
  "8": "strict-lenient",
  "9": "refined-rugged",
  "10": "trusting-suspicious",
  "11": "innocent-worldly",
  "12": "artistic-scientific",
  "13": "stoic-expressive",
  "14": "cunning-honorable",
  "15": "orderly-chaotic",
  "16": "normal-weird",
  "17": "competitive-cooperative",
  "18": "tense-relaxed",
  "19": "brave-careful",
  "20": "spiritual-skeptical",
  "21": "unlucky-fortunate",
  "22": "ferocious-pacifist",
  "23": "modest-flamboyant",
  "24": "dominant-submissive",
  "25": "forgiving-vengeful",
  "26": "wise-foolish",
  "27": "impulsive-cautious",
  "28": "loyal-traitorous",
  "29": "creative-conventional",
  "30": "curious-apathetic",
  "31": "rude-respectful",
  "32": "diligent-lazy",
  "33": "lustful-chaste",
  "34": "chatty-reserved",
  "35": "emotional-logical",
  "36": "moody-stable",
  "37": "dunce-genius",
  "38": "arrogant-humble",
  "39": "heroic-villainous",
  "40": "attractive-repulsive",
  "41": "rational-whimsical",
  "42": "mischievous-well behaved",
  "43": "aloof-obsessed",
  "44": "indulgent-sober",
  "45": "kinky-vanilla",
  "46": "straightforward-cryptic",
  "47": "spontaneous-deliberate",
  "48": "libertarian-socialist",
  "49": "scheduled-spontaneous",
  "50": "works hard-plays hard",
  "51": "reasoned-instinctual",
  "52": "focused on the present-focused on the future",
  "53": "empirical-theoretical",
  "54": "open-guarded",
  "55": "methodical-astonishing",
  "56": "mighty-puny",
  "57": "bossy-meek",
  "58": "barbaric-civilized",
  "59": "gregarious-private",
  "60": "quiet-loud",
  "61": "political-nonpolitical",
  "62": "confident-insecure",
  "63": "democratic-authoritarian",
  "64": "debased-pure",
  "65": "fast-slow",
  "66": "frugal-lavish",
  "67": "ludicrous-sensible",
  "68": "orange-purple",
  "69": "tall-short",
  "70": "young-old",
  "71": "down2earth-head@clouds",
  "72": "extrovert-introvert",
  "73": "open to new experinces-uncreative",
  "74": "calm-anxious",
  "75": "disorganized-self-disciplined",
  "76": "quarrelsome-warm",
  "77": "nerd-jock",
  "78": "lowbrow-highbrow",
  "79": "selfish-altruistic",
  "80": "autistic-neurotypical",
  "81": "angelic-demonic",
  "82": "hesitant-decisive",
  "83": "devout-heathen",
  "84": "cruel-kind",
  "85": "direct-roundabout",
  "86": "mathematical-literary",
  "87": "blue-collar-ivory-tower",
  "88": "slovenly-stylish",
  "89": "playful-shy",
  "90": "serious-bold",
  "91": "charming-trusting",
  "92": "awkward-suspicious",
  "93": "hipster-basic",
  "94": "coordinated-clumsy",
  "95": "funny-humorless",
  "96": "politically correct-edgy",
  "97": "rich-poor",
  "98": "hard-soft",
  "99": "remote-involved",
  "100": "metaphorical-literal",
  "101": "biased-impartial",
  "102": "mundane-extraordinary",
  "103": "tiresome-interesting",
  "104": "smooth-rough",
  "105": "spicy-mild",
  "106": "enslaved-emancipated",
  "107": "optimistic-pessimistic",
  "108": "sickly-healthy",
  "109": "luddite-technophile",
  "110": "vain-demure",
  "111": "high-tech-low-tech",
  "112": "flexible-rigid",
  "113": "cosmopolitan-provincial",
  "114": "arcane-mainstream",
  "115": "outlaw-sheriff",
  "116": "pronatalist-child free",
  "117": "sad-happy",
  "118": "jealous-compersive",
  "119": "bitter-sweet",
  "120": "resigned-resistant",
  "121": "sarcastic-genuine",
  "122": "human-animalistic",
  "123": "sporty-bookish",
  "124": "moderate-extreme",
  "125": "angry-good-humored",
  "126": "depressed-bright",
  "127": "self-conscious-self-assured",
  "128": "vulnerable-armoured",
  "129": "warm-cold",
  "130": "assertive-passive",
  "131": "active-slothful",
  "132": "imaginative-practical",
  "133": "adventurous-stick-in-the-mud",
  "134": "obedient-rebellious",
  "135": "competent-incompetent",
  "136": "unambitious-driven",
  "137": "simple-complicated",
  "138": "proletariat-bourgeoisie",
  "139": "alpha-beta",
  "140": "'right-brained'-'left-brained'",
  "141": "thick-skinned-sensitive",
  "142": "charismatic-uninspiring",
  "143": "feisty-gracious",
  "144": "eloquent-unpolished",
  "145": "high IQ-low IQ",
  "146": "insider-outsider",
  "147": "morning lark-night owl",
  "148": "thin-thick",
  "149": "sheeple-conspiracist",
  "150": "neat-messy",
  "151": "vague-precise",
  "152": "philosophical-real",
  "153": "modern-historical",
  "154": "judgemental-accepting",
  "155": "average-deviant",
  "156": "gossiping-confidential",
  "157": "official-backdoor",
  "158": "scholarly-crafty",
  "159": "leisurely-hurried",
  "160": "explorer-builder",
  "161": "captain-first-mate",
  "162": "mysterious-unambiguous",
  "163": "independent-codependent",
  "164": "family-first-work-first",
  "165": "scruffy-manicured",
  "166": "wild-tame",
  "167": "prestigious-disreputable",
  "168": "scandalous-proper",
  "169": "unprepared-hoarder",
  "170": "sheltered-street-smart",
  "171": "open-minded-close-minded",
  "172": "permanent-transient",
  "173": "dramatic-no-nonsense",
  "174": "apprentice-master",
  "175": "straight-queer",
  "176": "androgynous-gendered",
  "177": "repetitive-varied",
  "178": "patient-impatient",
  "179": "poisonous-nurturing",
  "180": "creepy-disarming",
  "181": "inspiring-cringeworthy",
  "182": "soulless-soulful",
  "183": "hard-soft",
  "184": "beautiful-ugly",
  "185": "domestic-industrial",
  "186": "juvenile-mature",
  "187": "idealist-realist",
  "188": "nihilist-existentialist",
  "189": "objective-subjective",
  "190": "theist-atheist",
  "191": "classical-avant-garde",
  "192": "utilitarian-decorative",
  "193": "generalist-specialist",
  "194": "multicolored-monochrome",
  "195": "complimentary-insulting",
  "196": "individualist-communal",
  "197": "equitable-hypocritical",
  "198": "traditional-unorthodox",
  "199": "workaholic-slacker",
  "200": "resourceful-helpless",
  "201": "crazy-sane",
  "202": "anarchist-statist",
  "203": "cool-dorky",
  "204": "important-irrelevant",
  "205": "noob-pro",
  "206": "deranged-reasonable",
  "207": "rural-urban",
  "208": "introspective-not introspective",
  "209": "city-slicker-country-bumpkin",
  "210": "western-eastern",
  "211": "mad-glad",
  "212": "social-reclusive",
  "213": "studious-goof-off",
  "214": "slugabed-go-getter",
  "215": "penny-pincher-overspender",
  "216": "liberal-conservative",
  "217": "unassuming-pretentious",
  "218": "persistent-quitter",
  "219": "hedonist-monastic",
  "220": "patriotic-unpatriotic",
  "221": "tactful-indiscreet",
  "222": "wholesome-salacious",
  "223": "joyful-miserable",
  "224": "zany-regular",
  "225": "alert-oblivious",
  "226": "feminist-sexist",
  "227": "racist-egalitarian",
  "228": "abstract-concrete",
  "229": "formal-intimate",
  "230": "resolute-wavering",
  "231": "deep-shallow",
  "232": "valedictorian-drop out",
  "233": "minimalist-pack rat",
  "234": "trash-treasure",
  "235": "&#129392;-&#128579;",
  "236": "&#129396;-&#129395;",
  "237": "&#128526;-&#129488;",
  "238": "&#128557;-&#128512;",
  "239": "&#129297;-&#129312;",
  "240": "&#128125;-&#129313;",
  "241": "&#128148;-&#128157;",
  "242": "&#129302;-&#128123;",
  "243": "&#128169;-&#127775;",
  "244": "&#128170;-&#129504;",
  "245": "&#128587;&zwj;&#9794;&#65039;-&#128581;&zwj;&#9794;&#65039;",
  "246": "&#128104;&zwj;&#9877;&#65039;-&#128104;&zwj;&#128295;",
  "247": "&#128105;&zwj;&#128300;-&#128105;&zwj;&#127908;",
  "248": "&#128520;-&#128519;",
  "249": "&#129300;-&#129323;",
  "250": "&#128024;-&#128000;",
  "251": "&#128046;-&#128055;",
  "252": "&#128052;-&#129412;",
  "253": "&#128041;-&#128018;",
  "254": "&#128692;-&#127947;&#65039;&zwj;&#9794;&#65039;",
  "255": "&#129338;-&#127948;",
  "256": "&#128131;-&#129493;",
  "257": "&#129497;-&#128104;&zwj;&#128640;",
  "258": "&#128016;-&#129426;",
  "259": "&#129415;-&#128063;",
  "260": "&#128556;-&#128527;",
  "261": "&#129296;-&#128540;",
  "262": "&#129315;-&#128522;",
  "263": "&#129495;-&#128716;",
  "264": "&#129406;-&#128095;",
  "265": "&#127913;-&#129506;",
  "266": "&#128200;-&#128201;",
  "267": "stinky-fresh",
  "268": "legit-scrub",
  "269": "self-destructive-self-improving",
  "270": "French-Russian",
  "271": "German-English",
  "272": "Italian-Swedish",
  "273": "Greek-Roman",
  "274": "traumatized-flourishing",
  "275": "sturdy-flimsy",
  "276": "macho-metrosexual",
  "277": "claustrophobic-spelunker",
  "278": "offended-chill",
  "279": "rhythmic-stuttering",
  "280": "musical-off-key",
  "281": "lost-enlightened",
  "282": "masochistic-pain-avoidant",
  "283": "efficient-overprepared",
  "284": "oppressed-privileged",
  "285": "sunny-gloomy",
  "286": "vegan-cannibal",
  "287": "loveable-punchable",
  "288": "slow-talking-fast-talking",
  "289": "believable-poorly-written",
  "290": "vibrant-geriatric",
  "291": "consistent-variable",
  "292": "dispassionate-romantic",
  "293": "linear-circular",
  "294": "intense-lighthearted",
  "295": "knowledgeable-ignorant",
  "296": "fixable-unfixable",
  "297": "exuberant-subdued",
  "298": "secretive-open-book",
  "299": "perceptive-unobservant",
  "300": "folksy-presidential",
  "301": "corporate-freelance",
  "302": "sleepy-frenzied",
  "303": "loose-tight",
  "304": "narcissistic-low self esteem",
  "305": "poetic-factual",
  "306": "melee-ranged",
  "307": "giggling-chortling",
  "308": "whippersnapper-sage",
  "309": "tailor-blacksmith",
  "310": "hunter-gatherer",
  "311": "experimental-reliable",
  "312": "moist-dry",
  "313": "trolling-triggered",
  "314": "tattle-tale-f***-the-police",
  "315": "punk rock-preppy",
  "316": "realistic-fantastical",
  "317": "trendy-vintage",
  "318": "factual-exaggerating",
  "319": "good-cook-bad-cook",
  "320": "comedic-dramatic",
  "321": "OCD-ADHD",
  "322": "interrupting-attentive",
  "323": "exhibitionist-bashful",
  "324": "badass-weakass",
  "325": "gamer-non-gamer",
  "326": "random-pointed",
  "327": "epic-deep",
  "328": "serene-pensive",
  "329": "bored-interested",
  "330": "envious-prideful",
  "331": "ironic-profound",
  "332": "sexual-asexual",
  "333": "&#129397;-&#129398;",
  "334": "&#127875;-&#128128;",
  "335": "&#127936;-&#127912;",
  "336": "clean-perverted",
  "337": "empath-psychopath",
  "338": "haunted-blissful",
  "339": "entitled-grateful",
  "340": "ambitious-realistic",
  "341": "stuck-in-the-past-forward-thinking",
  "342": "fire-water",
  "343": "earth-air",
  "344": "lover-fighter",
  "345": "overachiever-underachiever",
  "346": "Coke-Pepsi",
  "347": "twitchy-still",
  "348": "freak-normie",
  "349": "thinker-doer",
  "350": "hard-work-natural-talent",
  "351": "stingy-generous",
  "352": "stubborn-accommodating",
  "353": "extravagant-thrifty",
  "354": "demanding-unchallenging",
  "355": "two-faced-one-faced",
  "356": "plastic-wooden",
  "357": "neutral-opinionated",
  "358": "chivalrous-businesslike",
  "359": "high standards-desperate",
  "360": "on-time-tardy",
  "361": "everyman-chosen one",
  "362": "jealous-opinionated",
  "363": "protagonist-antagonist",
  "364": "devoted-unfaithful",
  "365": "fearmongering-reassuring",
  "366": "common sense-analysis",
  "367": "unemotional-emotional",
   "368": "rap-rock",
  "369": "genocidal-not genocidal",
  "370": "cat person-dog person",
  "371": "indie-pop",
  "372": "cultured-rustic",
  "373": "tautology-oxymoron",
  "374": "bad boy-white knight",
  "375": "princess-queen",
  "376": "hypochondriac-stoic",
  "377": "yes-man-contrarian",
  "378": "giving-receiving",
  "379": "chic-cheesy",
  "380": "celebrity-boy/girl-next-door",
  "381": "goth-flower child",
  "382": "summer-winter",
  "383": "frank-sugarcoated",
  "384": "naive-paranoid",
  "385": "gullible-cynical",
  "386": "motivated-unmotivated",
  "387": "radical-centrist",
  "388": "monotone-expressive",
  "389": "love-focused-money-focused",
  "390": "transparent-machiavellian",
  "391": "timid-cocky",
  "392": "concise-long-winded",
  "393": "picky-always down",
  "394": "proactive-reactive",
  "395": "prudish-flirtatious",
  "396": "innocent-jaded",
  "397": "touchy-feely-distant",
  "398": "muddy-washed",
  "399": "quirky-predictable",
  "400": "never cries-often crying",
  "401": "main character-side character",
  "402": "original-cliché",
  "403": "hugs-handshakes",
  "404": "homebody-world traveler",
  "405": "naughty-nice",
  "406": "junkie-straight edge",
  "407": "small-vocabulary-big-vocabulary",
  "408": "dystopian-utopian",
  "409": "parental-childlike",
  "410": "writer-reader",
  "411": "creator-consumer",
  "412": "capitalist-communist",
  "413": "positive-negative",
  "414": "grounded-fantasy-prone",
  "415": "thinker-feeler",
  "416": "insightful-generic",
  "417": "questioning-believing",
  "418": "proud-apologetic",
  "419": "bubbly-flat",
  "420": "tired-wired",
  "421": "woke-problematic",
  "422": "grumpy-cheery",
  "423": "hippie-militaristic",
  "424": "gluttonous-moderate",
  "425": "flawed-perfect",
  "426": "sweet-savory",
  "427": "annoying-unannoying",
  "428": "good-manners-bad-manners",
  "429": "evolutionist-creationist",
  "430": "fulfilled-unfulfilled",
  "431": "friendly-unfriendly",
  "432": "innovative-routine",
  "433": "delicate-coarse",
  "434": "resentful-euphoric",
  "435": "uptight-easy",
  "436": "blue-red",
  "437": "slumbering-insomniac",
  "438": "spirited-lifeless",
  "439": "outgoing-withdrawn",
  "440": "Hates PDA-Constant PDA",
  "441": "buffoon-charmer",
  "442": "sloppy-fussy",
  "443": "accurate-off target",
  "444": "harsh-gentle",
  "445": "clinical-heartfelt",
  "446": "inappropriate-seemly",
  "447": "smug-sheepish",
  "448": "fake-real",
  "449": "popular-rejected",
  "450": "catty-supportive",
  "451": "chronically single-serial dater",
  "452": "people-person-things-person",
  "453": "eager-reluctant",
  "454": "goal-oriented-experience-oriented",
  "455": "outdoorsy-indoorsy",
  "456": "divine-earthly",
  "457": "foodie-unenthusiastic about food",
  "458": "chill-sassy",
  "459": "glamorous-spartan",
  "460": "prankster-anti-prank",
  "461": "goofy-unfrivolous",
  "462": "noble-jovial",
  "463": "blessed-cursed",
  "464": "forward-repressed",
  "465": "entrepreneur-employee",
  "466": "quivering-unstirring",
  "467": "mechanical-natural",
  "468": "minds-own-business-snoops",
  "469": "prying-unmeddlesome",
  "470": "leader-follower",
  "471": "handy-can't-fix-anything",
  "472": "green thumb-plant-neglecter",
  "473": "activist-nonpartisan",
  "474": "photographer-physicist",
  "475": "lumberjack-mad-scientist",
  "476": "pointless-meaningful",
  "477": "focused-absentminded",
  "478": "bear-wolf",
  "479": "lion-zebra",
  "480": "kangaroo-dolphin",
  "481": "all-seeing-blind",
  "482": "engineerial-lawyerly",
  "483": "love shy-cassanova",
  "484": "disturbing-enchanting",
  "485": "maverick-conformist",
  "486": "social climber-nonconformist",
  "487": "social chameleon-strong identity",
  "488": "awkward-comfortable",
  "489": "sincere-irreverent",
  "490": "intuitive-analytical",
  "491": "cringing away-welcoming experience",
  "492": "stereotypical-boundary breaking",
  "493": "energetic-mellow",
  "494": "hopeful-fearful",
  "495": "likes change-resists change",
  "496": "manic-mild",
  "497": "old-fashioned-progressive",
  "498": "gross-hygienic",
  "499": "stable-unstable",
  "500": "overthinker-underthinker"
}
trait_default_dict = {int(key): value for key, value in full_default_trait_dict.items()}


import pickle
import os
import numpy as np
import argparse
import torch
import json
from tqdm import tqdm
from embeddings_analysis.utils import get_strong_correlations, plot_r2_scores, find_files_with_key_words

from pathlib import Path
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load each embedding dictionary

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

# Set directory
embedding_dir = "../data/structured_embeddings"

class Autoencoder(nn.Module):
    def __init__(self, input_dim=768, latent_dim=20, dropout=0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon, z

class MoralClassifier(nn.Module):
    """A classifier that can inject character information into BERT embeddings."""

    def __init__(self, base_model, latent_dim=768, inject_operation = "summation"):
        super().__init__()
        self.bert = base_model
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, 1)  # Binary classification (moral or not)
        self.operation = inject_operation

    def forward(self, input_ids, attention_mask, char_vec=None):
        """
        A forward function. This can be extended to support more operations.
        """

        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # [CLS] token

        if char_vec is not None:
            if self.operation == "summation":
                cls_embedding = cls_embedding + char_vec  # Inject character info

        cls_embedding = self.dropout(cls_embedding)
        logits = self.classifier(cls_embedding)
        return logits


def method_1(embeddings_data, ratings_data):
    latent_list, rating_list = [], []
    

    for movie in embeddings_data:
        for character in embeddings_data[movie]:
            latent = embeddings_data[movie][character]
            rating = ratings_data[movie][character]

            if latent is not None and rating is not None:
                latent_list.append(torch.tensor(latent))
                rating_list.append(torch.tensor(rating))

    latent_matrix = torch.stack(latent_list).numpy()
    rating_matrix = torch.stack(rating_list).numpy()

    num_latent = latent_matrix.shape[1]
    num_traits = rating_matrix.shape[1]
    correlation_matrix = np.zeros((num_latent, num_traits))

    for i in range(num_latent):
        for j in range(num_traits):
            x, y = latent_matrix[:, i], rating_matrix[:, j]
            correlation_matrix[i, j] = np.nan if np.std(x) == 0 or np.std(y) == 0 else spearmanr(x, y)[0]

    return pd.DataFrame(
        correlation_matrix,
        index=[f"latent_{i}" for i in range(num_latent)],
        columns=[f"trait_{j}" for j in range(num_traits)]
    )


def method_2(embeddings_data, ratings_data):
    latent_list, rating_list = [], []

    for movie in embeddings_data:
        for character in embeddings_data[movie]:
            latent = embeddings_data[movie][character]
            rating = ratings_data[movie][character]

            if latent is not None and rating is not None:
                latent_list.append(torch.tensor(latent))
                rating_list.append(torch.tensor(rating))

    X = torch.stack(latent_list).numpy()
    Y = torch.stack(rating_list).numpy()

    n_traits = Y.shape[1]
    latent_dim = X.shape[1]
    rows = []

    for j in range(n_traits):
        y = Y[:, j]
        if np.std(y) == 0:
            row = {"trait_index": f"trait_{j}", "r2_score": np.nan}
            row.update({f"latent_{k}": np.nan for k in range(latent_dim)})
        else:
            model = LinearRegression().fit(X, y)
            r2 = r2_score(y, model.predict(X))
            row = {"trait_index": f"trait_{j}", "r2_score": r2}
            row.update({f"latent_{k}": coef for k, coef in enumerate(model.coef_)})
        rows.append(row)

    return pd.DataFrame(rows)

def load_ratings(path:str):
    """Load ratings from the specified path.

    We're expecting the data to be in the form of "structured data" JSON files.
    """

    with open(path, 'r') as f:
        data = json.load(f)

    moral_ratings = {}

    for movie, movie_data in data.items():
        moral_ratings[movie] = {}
        for character, char_data in movie_data["characters"].items():
            if "rating" in char_data:
                moral_ratings[movie][character] = char_data["rating"]
            else:
                moral_ratings[movie][character] = None

    return moral_ratings


def load_model_and_tokenizer(model_name, model_path):
    base_model = AutoModel.from_pretrained(model_name)

    # Reconstruct full classifier model
    classifier = MoralClassifier(base_model)

    # Load state_dict from file
    classifier.load_state_dict(torch.load(model_path, map_location="cpu"))

    # Move to eval mode and device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier.to(device)
    classifier.eval()

    return classifier.bert, AutoTokenizer.from_pretrained(model_name)

def generate_embeddings(tokenizer, model, sentences, batch_size=32, max_length=256,
                        pooling="mean", exclude_special_tokens=True, to_numpy=True, device=None):
  
    """Generate sentence embeddings with a BERT-based model. """
    if device is None:
        device = next(model.parameters()).device
    out_chunks = []

    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i+batch_size]
        enc = tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length
        ).to(device)

        with torch.no_grad():
            h = model(**enc).last_hidden_state                         # (B, T, H)

            if pooling == "cls":
                pooled = h[:, 0, :]                                    # (B, H)
            else:
                attn = enc["attention_mask"].float()                   # (B, T)
                if exclude_special_tokens and "special_tokens_mask" in enc:
                    attn = attn * (1.0 - enc["special_tokens_mask"].float())
                attn = attn.unsqueeze(-1)                               # (B, T, 1)
                pooled = (h * attn).sum(1) / attn.sum(1).clamp(min=1e-9)

        out_chunks.append(pooled.cpu())
        del enc, h, pooled
        torch.cuda.empty_cache()

    out = torch.cat(out_chunks, dim=0)
    return out.numpy() if to_numpy else out


def recompute_embeddings(model_path, sentence_data, pooling, batch_size=1, model_name="bert-base-uncased"):
    """Load BERT-based embeddings or recompute them if necessary."""
    
    if pooling not in ["cls", "mean"]:
        raise ValueError("Invalid pooling method. Choose 'cls' or 'mean'.")
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_name, model_path)

    embeddings_dct = {}

    for movie, characters in tqdm(sentence_data["sentences"].items(), desc="Processing movies"):
        embeddings_dct[movie] = {}

        for character, sentences in tqdm(characters.items(), desc=f"Processing characters in {movie}", leave=False):
            embeddings_dct[movie][character] = generate_embeddings(tokenizer, model, sentences)
    return embeddings_dct 


def encode_latent_embeddings(embeddings, model_H):
    """Encode the averaged character embeddings using the AutoEncoder model."""
    if not isinstance(embeddings, dict):
        raise ValueError("Embeddings should be a dictionary with movie and character keys.")

    encoded_embeddings = {}
    device = next(model_H.parameters()).device  # Ensure tensor is on correct device

    for movie, characters in tqdm(embeddings.items(), desc="Encoding movies"):
        encoded_embeddings[movie] = {}
        for character, sentence_embeddings in tqdm(characters.items(), desc=f"Encoding characters in {movie}", leave=False):
            if len(sentence_embeddings) == 0:  # handle empty list
                print(f"⚠️ No embeddings for {character} in {movie}. Skipping.")
                continue
            sentence_tensor = torch.tensor(sentence_embeddings).float().to(device)  # shape: [N, 768]
            avg_embedding = sentence_tensor.mean(dim=0).unsqueeze(0)               # shape: [1, 768]
            with torch.no_grad():
                _, encoded = model_H(avg_embedding)
            encoded_embeddings[movie][character] = encoded.squeeze().cpu().numpy()  # shape: [latent_dim]

    return encoded_embeddings

import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_character_correlation_heatmap(character_embeddings, target_dir):
    # Step 1: flatten dict
    flat_data = {}
    for movie, chars in character_embeddings.items():
        for char, embed in chars.items():
            key = f"{movie}_{char}"
            # Ensure tensor → numpy
            flat_data[key] = embed.detach().cpu().numpy() if isinstance(embed, torch.Tensor) else embed
    
    # Step 2: DataFrame
    df = pd.DataFrame(flat_data).T  # shape: (num_chars, latent_dim)
    
    # Step 3: correlation matrix
    corr = df.corr()
    
    # Step 4: heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.index,
                cmap="coolwarm", center=0, annot=False)
    plt.title("Correlation Heatmap of Character Embeddings")
    plt.tight_layout()
    plt.savefig(os.path.join(target_dir, "character_correlation_heatmap.png"))

def plot_heatmap_pvals(corr_df, pval_df, target_dir, trait_dict=None, title="Spearman correlation p-values"):
    # Annotate with only p-values
    annot = pval_df.round(3).astype(str)

    plt.figure(figsize=(15,8))
    sns.heatmap(
        corr_df,             # colors come from correlations
        cmap="coolwarm",
        center=0,
        annot=annot,         # numbers = p-values
        fmt="",              # accept strings
        cbar_kws={'label': 'Spearman ρ'}  # colorbar still shows correlation scale
    )

    # Replace x-axis labels with trait names if trait_dict provided
    if trait_dict is not None:
        labels = [trait_dict[int(col.split('_')[1])] for col in corr_df.columns]
    else:
        labels = corr_df.columns
    plt.xticks(ticks=np.arange(len(labels)) + 0.5, labels=labels, rotation=45, ha='right')

    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(target_dir, "correlation_pvalues_heatmap.png"))

import pandas as pd
import numpy as np
import statsmodels.api as sm

def regression_analysis(X, Y):
    n_traits = Y.shape[1]
    latent_dim = X.shape[1]
    rows = []

    for j in range(n_traits):
        y = Y[:, j]
        if np.std(y) == 0:
            row = {"trait_index": f"trait_{j}", "r2_score": np.nan, "model_pvalue": np.nan}
            row.update({f"latent_{k}_pval": np.nan for k in range(latent_dim)})
        else:
            model = sm.OLS(y, sm.add_constant(X)).fit()
            r2 = model.rsquared
            row = {
                "trait_index": f"trait_{j}",
                "r2_score": r2,
                "model_pvalue": model.f_pvalue
            }
            # Add p-values for each latent dimension
            for k in range(latent_dim):
                row[f"latent_{k}_pval"] = model.pvalues[k+1]  # skip intercept
        rows.append(row)

    return pd.DataFrame(rows)

from scipy.stats import spearmanr

def compute_spearman(embeddings_data, ratings_data):
    latent_list, rating_list = [], []

    for movie in embeddings_data:
        for character in embeddings_data[movie]:
            if movie in ratings_data and character in ratings_data[movie]:
                latent = embeddings_data[movie][character]
                rating = ratings_data[movie][character]
                latent_list.append(torch.tensor(latent))
                rating_list.append(torch.tensor(rating))

    X = torch.stack(latent_list).numpy()
    Y = torch.stack(rating_list).numpy()

    n_latent, n_traits = X.shape[1], Y.shape[1]
    corr_matrix = np.zeros((n_latent, n_traits))
    pval_matrix = np.zeros((n_latent, n_traits))

    for i in range(n_latent):
        for j in range(n_traits):
            rho, pval = spearmanr(X[:, i], Y[:, j])
            corr_matrix[i, j] = rho
            pval_matrix[i, j] = pval

    corr_df = pd.DataFrame(corr_matrix, 
                           index=[f"latent_{i}" for i in range(n_latent)],
                           columns=[f"trait_{j}" for j in range(n_traits)])
    pval_df = pd.DataFrame(pval_matrix, 
                           index=[f"latent_{i}" for i in range(n_latent)],
                           columns=[f"trait_{j}" for j in range(n_traits)])
    return corr_df, pval_df


def show_top_traits_by_r2(regression_results, target_folder, top_n=10):
    # Sort the results by R² in descending order and select the top 50
    top_50_results = regression_results.nlargest(top_n, "r2_score")

    # Determine bar colors
    bar_colors = [
        "orange" if int(trait.split('_')[1]) in trait_dict else "gray"
        for trait in top_50_results["trait_index"]
    ]

    # Plot
    plt.figure(figsize=(12, 10))
    ax = sns.barplot(
        x="r2_score",
        y=top_50_results["trait_index"].apply(
            lambda x: trait_default_dict.get(int(x.split('_')[1]), x)
        ),
        data=top_50_results,
        palette=bar_colors
    )

    plt.xlabel("R² Score")
    plt.ylabel("Trait")
    plt.title(f"Top {top_n} Traits by R² Score")

    # Annotate with p-values
    for i, (r2, pval) in enumerate(zip(top_50_results["r2_score"], top_50_results["model_pvalue"])):
        ax.text(
            r2 + 0.01,  # a bit to the right of the bar
            i,          # y position (bar index)
            f"p={pval:.3g}",  # formatted p-value
            va="center"
        )

    plt.tight_layout()
    plt.savefig(os.path.join(target_folder, "top_traits_by_r2.png"))

def plot_r2_histogram(regression_df, target_dir, title="Distribution of R² scores"):
    plt.figure(figsize=(10,6))
    sns.histplot(regression_df['r2_score'].dropna(), bins=40, kde=True)
    plt.title(title)
    plt.xlabel("R² score")
    plt.ylabel("Frequency")
    plt.savefig(os.path.join(target_dir, "r2_score_histogram.png"))

def load_or_compute_embeddings(embeddings_path, sentence_data_path = None, recompute=False, pooling_method = "cls", model_name = "bert-base-uncased"):
    
    if not recompute:
        try:
            with open(embeddings_path, "rb") as f:
                embeddings = pickle.load(f)
                return embeddings
        except FileNotFoundError:
            try: 
                embeddings_file = os.path.join(embeddings_path, "latent_embeddings.pkl")
                with open(embeddings_file, "rb") as f:
                    embeddings = pickle.load(f)
                    return embeddings
            except FileNotFoundError:
                raise FileNotFoundError("Embeddings file not found. Please recompute embeddings or provide the correct path.")
    else:
        # We need to provide model_name to choose the correct tokenizer and model.
        # Here embeddings_path is a folder containing the model and tokenizer.
        # We need to find the model and tokenizer in the embeddings_path.
        target_folder = embeddings_path
        
        classifier_path = find_files_with_key_words(target_folder, "classifier")

        if classifier_path is None:
            raise FileNotFoundError("Classifier model not found in the specified path.")

        # Define the sentence data
        if sentence_data_path is None:
            raise ValueError("Please provide the path to the sentence data file when recomputing embeddings.")

        with open(sentence_data_path, "r") as f:
            sentence_data = json.load(f) 

        # embeddings contains the BERT-based embeddings.
        embeddings = recompute_embeddings(classifier_path, sentence_data, pooling_method, model_name = model_name)

        # Save the embeddings to a file
        with open(os.path.join(target_folder, "embeddings.pkl"), "wb") as f:
            pickle.dump(embeddings, f)
        
        print("Embeddings recomputed and saved.")

        # Load the AutoEncoder model (model_H) from the embeddings_path
        ae_path = find_files_with_key_words(target_folder, "model_H")

        if ae_path is None:
            raise FileNotFoundError("AutoEncoder model not found in the specified path.")
        
        model_H = Autoencoder()
        model_H.load_state_dict(torch.load(ae_path, map_location="cpu"))
        model_H.eval()
        model_H.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        latent_embeddings = encode_latent_embeddings(embeddings, model_H)

        # Save the embeddings to a file
        with open(os.path.join(target_folder, "latent_embeddings.pkl"), "wb") as f:
            pickle.dump(latent_embeddings, f)

        print("Latent embeddings computed and saved.")
        
        return latent_embeddings

def run_analysis(embeddings_path, output_dir, rating_data_path, sentence_data_path = None, recompute_embeddings = False, pooling_method = "cls", model_name = "bert-base-uncased", correlation_threshold=0.4):
    """This will load the data required, run the methods, and store the results.

    The results will be stored in the same folder as the embeddings data.
    
    Args:
        embeddings_path (str): Path to the folder containing the embeddings data or the BERT-based model that computes them.
        recompute_embeddings (bool): Whether to recompute embeddings.
    """

    embeddings_data = load_or_compute_embeddings(embeddings_path, sentence_data_path=sentence_data_path, recompute=recompute_embeddings, pooling_method=pooling_method, model_name=model_name)
    rating_data = load_ratings(rating_data_path)
    
    # Embeddings data here contains sentence-level embeddings for each character in each movie.
    # Before proceeding to analysis, we need to transform them into character-level emebddings. 
    method_1_result = method_1(embeddings_data, rating_data)
    method_2_result = method_2(embeddings_data, rating_data)

    # Save the results
    # if recompute_embeddings:
    #     method_1_result.to_csv(os.path.join(embeddings_path, "method_1_results.csv"))
    #     method_2_result.to_csv(os.path.join(embeddings_path, "method_2_results.csv"))
    #     save_folder_path = embeddings_path
    # else:
    #     # embeddings_path is a file path, so we save in the same directory
    #     method_1_result.to_csv(os.path.join(os.path.dirname(embeddings_path), "method_1_results.csv"))
    #     method_2_result.to_csv(os.path.join(os.path.dirname(embeddings_path), "method_2_results.csv"))
    #     save_folder_path = Path(embeddings_path).parent

    X = np.stack([
        embeddings_data[movie][char].numpy()
        for movie in embeddings_data
        for char in embeddings_data[movie]
    ])

    Y = np.stack([
        rating_data[movie][char]
        for movie in embeddings_data
        for char in embeddings_data[movie]
    ])

    save_folder_path = output_dir
    method_1_result.to_csv(os.path.join(save_folder_path, "method_1_results.csv"))
    method_2_result.to_csv(os.path.join(save_folder_path, "method_2_results.csv"))

    plot_r2_scores(
        method_2_result,
        top_n=10,
        figsize=(12, 6),
        title="Top 10 Trait-wise R² Scores",
        save_path = os.path.join(save_folder_path, "top_10_r2_scores.png")
    )

    get_strong_correlations(method_1_result, threshold=correlation_threshold, save_path=os.path.join(save_folder_path, "strong_correlations.csv"))

    plot_character_correlation_heatmap(embeddings_data, save_folder_path)

    corr_df, pval_df = compute_spearman(embeddings_data, rating_data)

    moral_traits = MORAL_TRAIT_COLS
    moral_corr_df = corr_df[moral_traits]
    moral_pval_df = pval_df[moral_traits]
    # Save the spearman correlation matrix along with its p-values

    # Replace row names in corr_df with corresponding trait names from trait_default_dict
    corr_df.rename(columns={f"trait_{i}": trait_default_dict.get(i + 1, f"trait_{i}") for i in range(len(corr_df.columns))}, inplace=True)
    pval_df.rename(columns={f"trait_{i}": trait_default_dict.get(i + 1, f"trait_{i}") for i in range(len(pval_df.columns))}, inplace=True)

    corr_df.to_csv(os.path.join(save_folder_path, "spearman_correlation_matrix.csv"), index=True)
    pval_df.to_csv(os.path.join(save_folder_path, "spearman_pvalue_matrix.csv"), index=True)

    plot_heatmap_pvals(moral_corr_df, moral_pval_df, trait_dict, title="Spearman correlations between latent dimensions and moral traits")

    # REGRESSION ANALYSIS

    regression_results = regression_analysis(X, Y)
    # Add a new column 'trait_name' as the first column
    regression_results.insert(
        0,
        'trait_name',
        regression_results['trait_index'].apply(
            lambda x: trait_default_dict.get(int(x.split('_')[1]) + 1, x)
        )
    )
    regression_results.to_csv(os.path.join(save_folder_path, "regression_results.csv"), index=False)

    show_top_traits_by_r2(regression_results, save_folder_path, top_n=10)
    plot_r2_histogram(regression_results, save_folder_path, title="Distribution of R² scores from regression analysis")

    return method_1_result, method_2_result

def main(args):

    run_analysis(
        embeddings_path=args.source_folder_path,
        sentence_data_path=args.sentence_data_path,
        recompute_embeddings=args.recompute_embeddings,
        model_name=args.model_name,
        pooling_method="cls" if args.cls_embeddings else "mean",
        rating_data_path=args.rating_data_path,
        correlation_threshold=args.correlation_threshold,
        output_dir=args.output_dir
    )

if __name__ == "__main__":
    print("Starting Embeddings analysis...")

    parser = argparse.ArgumentParser(description="Example CLI script")
    
    # Positional argument

    # This needs to be a folder if we want to recompute embeddings and a file if we want to load them.
    # We let the user specify because unlike moral rating data, there are several embedding folders depending
    # on the training trials. 
    parser.add_argument("source_folder_path", help="The path to the source folder")
    # Optional argument with default
    # parser.add_argument("--trainable_base", type=int, default=1, help="Whether the model is trainable or not (1 for yes, 0 for no)")
    parser.add_argument(
        "--model_name",
        type=str,
        default="bert-base-uncased",
        help="The name of the base model to use for embeddings"
    )
    # Flag (boolean switch)
    parser.add_argument("--recompute_embeddings", action="store_true", help="The base model is trainable")
    # Currently we only support CLS and mean-pooled embeddings.
    parser.add_argument("--cls_embeddings", action="store_true", help="Use CLS embeddings for injection")
    parser.add_argument("--sentence_data_path", type=str, help="Path to the sentence data file")
    parser.add_argument("--rating_data_path", type=str, help="Path to the character ratings data file")
    parser.add_argument("--correlation_threshold", type=float, default=0.4, help="Threshold for strong correlations")
    parser.add_argument("--output_dir", type=str, help="Directory to save the analysis results")

    # Parse the arguments
    args = parser.parse_args()  

    main(args)
    print("Embeddings analysis completed.")