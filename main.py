import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from music21 import converter, instrument, note, stream
from music21.midi import MidiException

def preprocess_midi_files(directory):
    all_notes = []
    for file in os.listdir(directory):
        if file.endswith(".mid"):
            file_path = os.path.join(directory, file)
            print(f"Processing {file}")
            try:
                notes = preprocess_midi(file_path)
                all_notes.extend(notes)
            except Exception as e:
                print(f"Error processing {file}: {str(e)}")
                continue
    if not all_notes:
        raise ValueError("No valid MIDI files were processed. Please check your MIDI files and directory path.")
    return all_notes

def preprocess_midi(file_path):
    try:
        midi = converter.parse(file_path)
    except MidiException as e:
        print(f"MidiException: {str(e)}")
        raise
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        raise

    notes = []
    try:
        for element in midi.flat.notesAndRests:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, note.Rest):
                notes.append('Rest')
    except Exception as e:
        print(f"Error extracting notes: {str(e)}")
        raise

    if not notes:
        raise ValueError("No notes were extracted from the MIDI file.")
    
    return notes

def create_sequences(notes, sequence_length):
    pitchnames = sorted(set(notes))
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))
    network_input = []
    network_output = []
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])
        network_output.append(note_to_int[sequence_out])
    n_patterns = len(network_input)
    network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))
    network_input = network_input / float(len(pitchnames))
    network_output = tf.keras.utils.to_categorical(network_output)
    return (network_input, network_output), pitchnames

def create_model(network_input, n_vocab):
    model = Sequential()
    model.add(LSTM(
        256,
        input_shape=(network_input.shape[1], network_input.shape[2]),
        return_sequences=True
    ))
    model.add(LSTM(128, return_sequences=True))
    model.add(LSTM(64))
    model.add(Dense(n_vocab, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    return model

def generate_notes(model, network_input, pitchnames, n_vocab, num_notes=100):
    start = np.random.randint(0, len(network_input)-1)
    int_to_note = dict((number, note) for number, note in enumerate(pitchnames))
    pattern = network_input[start]
    prediction_output = []
    
    for note_index in range(num_notes):
        prediction_input = np.reshape(pattern, (1, len(pattern), 1))
        prediction_input = prediction_input / float(n_vocab)
        prediction = model.predict(prediction_input, verbose=0)
        index = np.argmax(prediction)
        result = int_to_note[index]
        prediction_output.append(result)
        pattern = np.append(pattern, index)
        pattern = pattern[1:len(pattern)]
    
    return prediction_output

def create_midi(prediction_output, file_name='output.mid'):
    offset = 0
    output_notes = []
    for pattern in prediction_output:
        if pattern != 'Rest':
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)
        else:
            new_note = note.Rest()
            new_note.offset = offset
            output_notes.append(new_note)
        offset += 0.5
    midi_stream = stream.Stream(output_notes)
    midi_stream.write('midi', fp=file_name)

midi_directory = './midi_files'
try:
    notes = preprocess_midi_files(midi_directory)
    sequence_length = 100
    (network_input, network_output), pitchnames = create_sequences(notes, sequence_length)
    n_vocab = len(set(notes))
    model = create_model(network_input, n_vocab)
    model.fit(network_input, network_output, epochs=50, batch_size=64)

    prediction_output = generate_notes(model, network_input, pitchnames, n_vocab, num_notes=500)
    create_midi(prediction_output, 'ai_generated_music.mid')
except Exception as e:
    print(f"An error occurred: {str(e)}")