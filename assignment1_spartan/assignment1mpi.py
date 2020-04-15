# -----------------------------------------------------------------------------
# COMP90024 Cluster and Cloud Computing Semester 1, 2020 - Assignment 1
# -----------------------------------------------------------------------------

import sys
import ijson, json
import re
import math
import itertools
import io
from collections import defaultdict, OrderedDict
from mpi4py import MPI

# -----------------------------------------------------------------------------

PARTITION = '\n#-------------------------------------------------------------\n'

# Setting up MPI parameters.
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
read = MPI.MODE_RDONLY

# int dicts to count hashtags and languages used.
hashtag_dict = defaultdict(int)
lang_dict = defaultdict(int)

if rank == 0:
    # Language code json, compiled according to the lang's section from
    # https://developer.twitter.com/en/docs/tweets/rules-and-filtering/overview/premium-operators
    
    # Load a json file containing language codes, if available in the same
    # directory. Skip otherwise.
    try:
        with open('languageCodes.json') as lang_code_json:
            LANG_CODES = json.loads(lang_code_json.read())
        print('Language code json loaded successfully. Continue...\n')

    except:
        print('Language code json undetected. Continue...\n')
        pass

    print('Number of workers: ' + str(size) +'.')

# -----------------------------------------------------------------------------

def hashtags_from_text(tweet_text):
    ''' 
    This function takes a tweet's text and return its hashtag(s) as a set, 
    if any. Returns an empty set otherwise.
    '''

    # Retain only substrings that comes after '#' and before punctuation, 
    # except for underscore, to get hashtags. Ignore non-ASCII hashtags.
    hashtags = ['#' + hashtag.lower() for hashtag in 
                re.findall(r'#(\w+)', tweet_text) if hashtag.isascii()]

    # Return unique hashtags found.
    return set(hashtags)

def combine_dict(dict_list, dict_type):
    ''' 
    This function takes a list of dictionaries and a defaultdict type
    and return the combined dictionaries as a single defaultdict of that type.
    '''
    combined_dict = defaultdict(dict_type)
    for dictionary in dict_list:
        for key, value in dictionary.items():
            combined_dict[key] += (value)

    return combined_dict

def scoreboard(input_dict, n, title, reverse_flag, lang_code_dict = None):
    ''' 
    This function takes a dictionary and return the top `n` keys 
    based on their values. Ties will have the same ranking.
    The scoreboard will have more than `n` entries in case more than one
    keys are ranked nth.
    title: a string containing the title of the scoreboard.
    reverse_flag (boolean): `False` for ascending order, `True` for descending.
    lang_code_dict: a json dict containing language codes (Optional).
    '''

    print(PARTITION)

    # Sort the dictionary.
    sorted_dict = sorted(input_dict.items(), key = lambda x: x[1], 
                        reverse = reverse_flag)
    sorted_dict = OrderedDict(sorted_dict)

    # Current rank to be printed.
    current_rank = 1

    # Rank and value of the previous key, used to handle ties.
    # Their initial values are that of the first key.
    prev_val = list(sorted_dict.items())[0][1]
    prev_rank = 1

    # Print the scoreboard entry by entry.
    print(title +'\n')
    for key, value in sorted_dict.items():

        # No more last place ties, return.
        if current_rank > n and value != prev_val:
            return

        # Use the previous rank in case of ties.
        if value == prev_val:

            # Print language name if there is a language dict provided.
            if lang_code_dict:
                if key in lang_code_dict.values():
                    for language, code in lang_code_dict.items():
                        if key == code:
                            print('{0}. {1} ({2}), {3}.'.format(
                                  str(prev_rank), language, key, value))

                # Language not defined in json file.
                else:
                    print('{0}. Undefined ({1}), {2}.'.format(str(prev_rank),
                          key, value))
            
            # No language dict provided.
            else:
                print('{0}. {1}, {2}.'.format(str(prev_rank), key, value))
        
        # Otherwise use current rank.
        else:

            # Print language name if there is a language dict provided.
            if lang_code_dict:
                if key in lang_code_dict.values():
                    for language, code in lang_code_dict.items():
                        if key == code:
                            print('{0}. {1} ({2}), {3}.'.format(
                                  str(current_rank), language, key, value))
                else:
                    # Language not defined in json file.
                    print('{0}. Undefined ({1}), {2}.'.format(str(current_rank),
                          key, value))

            # No language dict provided.
            else:
                print('{0}. {1}, {2}.'.format(str(current_rank), key, value))
            
            prev_rank = current_rank
        
        # Proceed to the next entry.
        prev_val = value
        current_rank += 1

# -----------------------------------------------------------------------------

# Take name of the file to be processed from the command line. 
# Program will exit if input file is not specified.
if (len(sys.argv) > 1):
    file_name = sys.argv[1]
else:
    if rank == 0:
        sys.exit('No json file specified. Please try again.')
    sys.exit()

# -----------------------------------------------------------------------------
# Read the file and get its size in byte.
read_file = MPI.File.Open(comm, file_name, read)
file_size = MPI.File.Get_size(read_file)
      
# Add some extra memory to buffer to avoid breaking the json structure.
# i.e some data in the end of one chunk and some data at the start of 
# the next chunk will overlap.
# Choose 20KB to be on the safe side.
overlap_size = 20480

# Buffer size for each worker.
buffer_size = int(math.ceil(file_size/size))
    
# Each worker will start reading at their respective offset.
worker_offset = buffer_size * rank

# Have each worker read one chunk of their assigned part at a time to 
# prevent integer overflow. Adjust value as needed.
if size < 2:
    chunk_num = 128
else:
    chunk_num = 32

# Size of each chunk in bytes.
chunk_size = int(math.ceil(buffer_size/chunk_num))

for i in range(chunk_num):
        
    chunk_offset = worker_offset + chunk_size * i

    # Read each chunk and overlapping data.
    chunk_buffer = bytearray(chunk_size + overlap_size)
    read_file.Read_at_all(chunk_offset, chunk_buffer)

    # Overlapping data only.
    overlap_buffer = bytearray(overlap_size)
    read_file.Read_at_all(chunk_offset + chunk_size, overlap_buffer)
    
    # Convert data in buffers to string.
    chunk_string = chunk_buffer.decode('utf-8', 'ignore').strip('\x00')
    overlap_string = overlap_buffer.decode('utf-8', 'ignore').strip('\x00')

    # Free memory.
    chunk_buffer = overlap_buffer = None

    # Find the index position where the overlapped data starts in the whole
    # data.
    overlap_index = chunk_string.rfind(overlap_string)

    # Adjusting chunk to process accounting for the overlapped data.
    # Each chunk  will begin after the first new line character
    # from the start...
    chunk_start = chunk_string.index('\n') + 1

    # ... and each chunk except for the last one of the last worker
    # will stop at the first new line character in the overlap region,
    # thus we will avoid having to process a tweet broken by splitting data.
    if rank == size - 1 and i == chunk_num - 1:
        chunk_string = chunk_string[chunk_start: ]

        # Fix the json chunk if necessary.
        try:
            json.load(chunk_string)
        except:
            chunk_string = '{"rows":[\n' + chunk_string[:-3] + ']}'

    else:
        chunk_end = overlap_index + overlap_string.index('\n')
        chunk_string = chunk_string[chunk_start: chunk_end]

        # Fix the json chunk if necessary.
        try:
            json.load(chunk_string)
        except:
            chunk_string = '{"rows":[\n' + chunk_string[:-2] + ']}'

    # Parse json data.
    parser = ijson.parse(io.StringIO(chunk_string))
    try:
        for prefix, event, value in parser:

            # Extract hashtags from tweet's text.
            if prefix == 'rows.item.doc.text':
                hashtags = hashtags_from_text(value)
                            
                # Increment extracted hashtags'.
                for hashtag in hashtags:
                    hashtag_dict[hashtag] += 1
                        
            # Increment language's count.
            elif prefix == 'rows.item.doc.metadata.iso_language_code':
                lang_dict[value] += 1
    
    # Skip any trailing bytes at the end resulted from the splitting process.
    except:
        pass

# Close the file after reading.
read_file.Close()

comm.Barrier()

# Gather the dictionaries from all worker.
all_hashtag_dicts = comm.gather(hashtag_dict, root = 0)
all_lang_dicts = comm.gather(lang_dict, root = 0)

comm.Barrier()

# At master, combine all dictionaries of each type into one.
if rank == 0:
        
    # Combine hashtag dictionaries.
    combined_hashtag_dict = combine_dict(all_hashtag_dicts, int)

    # Combine language dictionaries.
    combined_lang_dict = combine_dict(all_lang_dicts, int)
        
    # The lowest rank to be displayed on scoreboard.
    N = 10

    # Print the top `N` hashtags w/ counts.
    title = 'Top {0} hashtags.'.format(N)
    scoreboard(combined_hashtag_dict, N, title, True)

    #Print the top  `N` languages w/ counts.
    title = 'Top {0} languages.'.format(N)
    scoreboard(combined_lang_dict, N, title, True, LANG_CODES)

# -----------------------------------------------------------------------------
