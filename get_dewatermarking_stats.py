from bert_dewatermarking import iterated_replacements, tokenize_replacement, iterated_replacements_r
from generation_and_recovery import generate_with_blacklist, generate_without_blacklist, get_watermark_adherences, z_statistic_watermark 
import pandas as pd

prompt = "You don't understand! I coulda been something "
all_s = []
all_sr = []
all_z = []
all_zr = []

V = 2
PROMPT = ""
PROMPT = "_newprompt_2"
PROMPT = "_newprompt_3"
generations = list(pd.read_csv(f"dewater_stats_v1{PROMPT}.csv")["S"])

for count in range(50):
    s = generations[count] #generate_with_blacklist(prompt, max_tokens = 10)
    z = z_statistic_watermark(s)
    
    sr = iterated_replacements(s, 10)
    zr = z_statistic_watermark(sr)
    all_s.append(s)
    all_z.append(z)
    all_sr.append(sr)
    all_zr.append(zr)
    
pd.DataFrame({"S": all_s, "SR": all_sr, "Z": all_z, "ZR": all_zr}).to_csv(f"dewater_stats_v{V}{PROMPT}.csv")