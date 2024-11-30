import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup


from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2Model, GPT2Tokenizer
model = GPT2Model.from_pretrained("gpt2", output_hidden_states = True)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
model.resize_token_embeddings(len(tokenizer))


def matching(l1, l2, ind_l1):
    pointer_l1 = 0
    pointer_l2 = 0
    found = []
    iterations = 0
    while pointer_l1 < ind_l1 and iterations < 50:
        iterations += 1
        word1 = l1[pointer_l1].strip()
        word2 = l2[pointer_l2].strip()
        #print(word1)
        #print(word2)
        #print("---")
        if word1 == word2:
            pointer_l1 += 1
            pointer_l2 += 1
        elif word2 in word1:
            while word2 in word1:
                pointer_l2 += 1
                word2 = l2[pointer_l2]
            pointer_l1 += 1
        elif word1 in word2:
            while word1 in word2:
                pointer_l1 += 1
                word1 = l1[pointer_l1]
            pointer_l2 += 1
        else:
            found_option1 = matching(l1, l2[1:], ind_l1 - pointer_l1)
            found_option2 = matching(l1[1:], l2, ind_l1 - pointer_l1 - 1)
            found_option1 = [x + pointer_l1 + 2 for x in found_option1]
            found_option2 = [x + pointer_l1 + 2 for x in found_option2]
            return found_option1, found_option2
    word1 = l1[pointer_l1].strip()
    word2 = l2[pointer_l2].strip()
    while word2 in word1:
        found.append(pointer_l2)
        pointer_l2 += 1
        word2 = l2[pointer_l2]
    return min(found), max(found)

def parse_coinco_with_attributes(xml_file):
    """
    Parses the CoInCo XML file to extract sentences, tokens, and substitution attributes.

    Args:
        xml_file (str): Path to the CoInCo XML file.

    Returns:
        list of dict: A list of dictionaries containing sentence contexts, tokens, and substitution details.
    """
    # Parse the XML file
    tree = ET.parse(xml_file)
    root = tree.getroot()

    data = []
    
    coinco_data_points = []

    # Iterate over all `sent` elements
    for sent in root.findall('sent'):
        # Extract sentence context
        precontext = sent.find('precontext').text if sent.find('precontext') is not None else ""
        targetsentence = sent.find('targetsentence').text if sent.find('targetsentence') is not None else ""
        postcontext = sent.find('postcontext').text if sent.find('postcontext') is not None else ""

        full_sentence = f"{precontext.strip()} {targetsentence.strip()} {postcontext.strip()}".strip()

        tokens_data = []
        last_token = 0
        # Process tokens
        tokens = sent.find('tokens')
        token_details = "a"
        if tokens is not None:
            list_tokens = [token.attrib.get('wordform', None) for token in tokens.findall('token')]
            for i, token in enumerate(tokens.findall('token')):
                # Extract token details
                token_details = {
                    'id': token.attrib.get('id', None),
                    'wordform': token.attrib.get('wordform', None),
                    'lemma': token.attrib.get('lemma', None),
                    'posMASC': token.attrib.get('posMASC', None),
                    'posTT': token.attrib.get('posTT', None),
                    'problematic': token.attrib.get('problematic', 'no'),  # Default to 'no' if not specified
                    'i': i
                }
                # Process substitutions if available
                substitutions_element = token.find('substitutions')
                if substitutions_element is not None:
                    substitutions = []
                    for subst in substitutions_element.findall('subst'):
                        # Extract substitution attributes
                        substitution_details = {
                            'lemma': subst.attrib.get('lemma', None),
                            'pos': subst.attrib.get('pos', None),
                            'freq': int(subst.attrib.get('freq', 0))  # Convert freq to integer
                        }
                        substitutions.append(substitution_details['lemma'])
                    token_details['substitutions'] = substitutions
                    coinco_data_points.append({"sentence": full_sentence, "list_tokens": list_tokens, "token_number": token_details['i'], "subs": substitutions})
                tokens_data.append(token_details)
        
        last_token = token_details['id']

        # Store the result
        data.append({
            'full_sentence': full_sentence,
            'tokens': tokens_data
        })
        
    bad_apples = []
    for i, dp in enumerate(coinco_data_points):
        l = dp['list_tokens']
        sentence = " ".join(l)
        index = dp['token_number']
        inputs = tokenizer(sentence.replace("-",""), return_tensors="pt", add_special_tokens=False)
        l1 = l
        l2 = [tokenizer.decode([a]) for a in inputs['input_ids'][0]]
        try:
            matching(l1, l2, index)
        except:
            bad_apples.append(i)
            
    coinco_data_points = [coinco_data_points[i] for i in range(len(coinco_data_points)) if i not in bad_apples]


    return data, coinco_data_points

def parse_coinco_with_attributes_and_make_substitutions(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    data = []
    
    coinco_data_points = []

    # Iterate over all `sent` elements
    for sent in root.findall('sent'):
        # Extract sentence context
        precontext = sent.find('precontext').text if sent.find('precontext') is not None else ""
        targetsentence = sent.find('targetsentence').text if sent.find('targetsentence') is not None else ""
        postcontext = sent.find('postcontext').text if sent.find('postcontext') is not None else ""

        full_sentence = f"{precontext.strip()} {targetsentence.strip()} {postcontext.strip()}".strip()

        tokens_data = []
        last_token = 0
        # Process tokens
        tokens = sent.find('tokens')
        token_details = "a"
        if tokens is not None:
            list_tokens = [token.attrib.get('wordform', None) for token in tokens.findall('token')]
            for i, token in enumerate(tokens.findall('token')):
                # Extract token details
                token_details = {
                    'id': token.attrib.get('id', None),
                    'wordform': token.attrib.get('wordform', None),
                    'lemma': token.attrib.get('lemma', None),
                    'posMASC': token.attrib.get('posMASC', None),
                    'posTT': token.attrib.get('posTT', None),
                    'problematic': token.attrib.get('problematic', 'no'),  # Default to 'no' if not specified
                    'i': i
                }
                # Process substitutions if available
                substitutions_element = token.find('substitutions')
                if substitutions_element is not None:
                    substitutions = []
                    substitution_versions = []
                    for subst in substitutions_element.findall('subst'):
                        # Extract substitution attributes
                        substitution_details = {
                            'lemma': subst.attrib.get('lemma', None),
                            'pos': subst.attrib.get('pos', None),
                            'freq': int(subst.attrib.get('freq', 0))  # Convert freq to integer
                        }
                        #print(token_details['lemma'])
                        #print(substitution_details['lemma'])
                        subs_version = full_sentence.replace(token_details['wordform'], substitution_details['lemma'])
                        coinco_data_points.append({"sentence": full_sentence, "list_tokens": list_tokens, "token_number": token_details['i'], "subs": substitutions, "sub_version": subs_version})
                tokens_data.append(token_details)
        
        last_token = token_details['id']

        # Store the result
        data.append({
            'full_sentence': full_sentence,
            'tokens': tokens_data
        })
        
    bad_apples = []
    for i, dp in enumerate(coinco_data_points):
        l = dp['list_tokens']
        sentence = " ".join(l)
        index = dp['token_number']
        inputs = tokenizer(sentence.replace("-",""), return_tensors="pt", add_special_tokens=False)
        l1 = l
        l2 = [tokenizer.decode([a]) for a in inputs['input_ids'][0]]
        try:
            matching(l1, l2, index)
        except:
            bad_apples.append(i)
            
    coinco_data_points = [coinco_data_points[i] for i in range(len(coinco_data_points)) if i not in bad_apples]


    return data, coinco_data_points

_, cdp = parse_coinco_with_attributes_and_make_substitutions('coinco.xml')