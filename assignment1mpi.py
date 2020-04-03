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
PARTITION = '# ----------------------------------------------------------------'

# -----------------------------------------------------------------------------

# Setting up MPI parameters.
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
read = MPI.MODE_RDONLY
chunk_size = 8

# int dicts to count hashtags and languages used.
hashtag_dict = defaultdict(int)
lang_dict = defaultdict(int)

print('Number of available workers: ' + str(size) +'.')

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

def scoreboard(input_dict, n, reverse_flag):
    ''' 
    This function takes a dictionary and return the top `n` keys 
    based on their values. Ties will have the same ranking.
    The scoreboard will have more than `n` entries in case more than one
    keys are ranked nth.
    reverse_flag (boolean): `False` for ascending order, `True` for descending.
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
    for key, value in sorted_dict.items():

        # No more last place ties, return.
        if current_rank > n and value != prev_val:
            print(PARTITION + '\n')
            return

        # Use the previous rank in case of ties.
        if value == prev_val:
            print('{0}. {1}, {2}.'.format(str(prev_rank), key, value))
        
        # Otherwise use current rank.
        else:
            print('{0}. {1}, {2}.'.format(str(current_rank), key, value))
            prev_rank = current_rank
        
        prev_val = value
        current_rank += 1

# -----------------------------------------------------------------------------

# Take name of the file to be processed from the command line. 
# Default to 'tinyTwitter.json' if not provided.
if (len(sys.argv) > 1):
    file_name = sys.argv[1]
else:
    print('No filename specified. Default to `tinyTwitter.json`.')
    file_name = 'tinyTwitter.json'

# -----------------------------------------------------------------------------

# Size < 2 i.e single worker, read and process file serially.
if size < 2:
    
    print('Starting serialised version of the application...')
    print(PARTITION)

    with open(file_name, 'r') as input_file:
        parser = ijson.parse(input_file)
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
        except:
            pass

    # Print the top `N` hashtags w/ counts.
    N = 10
    print('Top {0} hashtags.'.format(N))
    scoreboard(hashtag_dict, N, True)

    #Print the top `N` languages w/ counts.
    N = 10
    print('Top {0} languages.'.format(N))
    scoreboard(lang_dict, N, True)


# -----------------------------------------------------------------------------

# Multiple workers, start the paralellised version.
else:
    print('Core ' + str(rank) + ' of ' + str(size) + ' starting...')
    print(PARTITION)

    # Read the file and get its size in byte.
    read_file = MPI.File.Open(comm, file_name, read)
    file_size = MPI.File.Get_size(read_file)

    # Buffer size for each worker.
    buffer_size = int(math.ceil(file_size/size))

    # Add some extra memory to buffer to avoid breaking the json structure.
    # i.e some data in the end of one chunk and some data at the start of 
    # the next chunk will overlap.
    # Choose 20KB to be on the safe side.
    overlap_size = 20480
    
    # Create overlap buffer to hold overlap data.
    overlap_buffer = bytearray(overlap_size)

    # Create buffer array to hold data.
    # The last worker does not need the extra memory at the end.
    if rank != size - 1:
        buffer = bytearray(buffer_size + overlap_size)
    else:
        buffer = bytearray(buffer_size)

    # Each worker will start reading at their respective offset.
    offset = buffer_size * rank
    print('rank ' + str(rank) + ' offset: ' + str(offset))

    # Each worker's total data (assigned part + overlapped data).
    read_file.Read_at_all(offset, buffer)

    # Each worker's overlapped data only.
    read_file.Read_at_all(offset + buffer_size, overlap_buffer)

    # Convert data in buffers to string.
    whole_string = buffer.decode('utf-8').strip('\x00') 
    overlap_string = overlap_buffer.decode('utf-8').strip('\x00')

    # Free the buffers to avoid memory error.
    buffer = overlap_buffer = None

    # Find the index position where the overlapped data starts in the whole
    # data.
    overlap_index = whole_string.find(overlap_string)

    # Adjusting chunk to be read accounting for the overlapped data.
    # Each worker except for the first one will begin after the first new line 
    # character from the start...
    chunk_start = whole_string.index('\n') + 1
    
    # ... and each worker except for the last one will stop at the 
    # first new line character in the overlap region. This way we will avoid
    # having to process a tweet broken by the splitting data among workers.
    if rank != size - 1:
        chunk_end = overlap_index + overlap_string.index('\n')
        whole_string = whole_string[chunk_start: chunk_end]
        
    else:
        whole_string = whole_string[chunk_start: ]

    comm.Barrier()

    # Reading is completed, close the file.
    read_file.Close()

    # Commence processing data.
    # Fix the string so that they are in correct json format.
    if rank != size - 1:
        whole_string = '{"rows":[' + whole_string[:-2] + ']}'
    else:
        whole_string = '{"rows":[' + whole_string[:-3] + ']}'

    # Parse json data.
    parser = ijson.parse(io.StringIO(whole_string))

    # Arrays to hold hashtags and languages.
    
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

        # Print the top `N` hashtags w/ counts.
        N = 10
        print('Top {0} hashtags.'.format(N))
        scoreboard(combined_hashtag_dict, N, True)

        #Print the top `N` languages w/ counts.
        N = 10
        print('Top {0} languages.'.format(N))
        scoreboard(combined_lang_dict, N, True)

# -----------------------------------------------------------------------------
