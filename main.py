import os
import numpy as np
import tensorflow as tf
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from music21 import converter, instrument, note, stream, chord
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def preprocess_midi_files(directory):
    all_notes = []
    for file in os.listdir(directory):
        if file.endswith(".mid"):
            file_path = os.path.join(directory, file)
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
        raise
    except Exception as e:
        raise

    notes = []
    try:
        for element in midi.flat.notesAndRests:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))
            elif isinstance(element, note.Rest):
                notes.append('Rest')
    except Exception as e:
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
    return (network_input, network_output), pitchnames, note_to_int

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

def generate_notes(model, network_input, pitchnames, n_vocab, seed_sequence, num_notes=100):
    int_to_note = dict((number, note) for number, note in enumerate(pitchnames))
    pattern = seed_sequence
    prediction_output = []
    
    for note_index in range(num_notes):
        prediction_input = np.reshape(pattern, (1, len(pattern), 1))
        prediction_input = prediction_input / float(n_vocab)
        prediction = model.predict(prediction_input, verbose=0)
        index = np.argmax(prediction)
        result = int_to_note[index]
        prediction_output.append(result)
        pattern = np.append(pattern[1:], index)
    
    return prediction_output

def create_midi(prediction_output, file_name='output.mid'):
    offset = 0
    output_notes = []
    for pattern in prediction_output:
        if pattern != 'Rest':
            if '.' in pattern:  # It's a chord
                notes_in_chord = pattern.split('.')
                chord_notes = []
                for current_note in notes_in_chord:
                    new_note = note.Note(int(current_note))
                    new_note.storedInstrument = instrument.Piano()
                    chord_notes.append(new_note)
                new_chord = chord.Chord(chord_notes)
                new_chord.offset = offset
                output_notes.append(new_chord)
            else:  # It's a note
                new_note = note.Note(pattern)
                new_note.offset = offset
                new_note.storedInstrument = instrument.Piano()
                output_notes.append(new_note)
        else:  # It's a rest
            new_note = note.Rest()
            new_note.offset = offset
            output_notes.append(new_note)
        offset += 0.5
    midi_stream = stream.Stream(output_notes)
    midi_stream.write('midi', fp=file_name)

def prompt_to_seed(prompt, note_to_int, sequence_length):
    seed_sequence = [note_to_int.get(note, 0) for note in prompt.split()]
    if len(seed_sequence) < sequence_length:
        seed_sequence = [0] * (sequence_length - len(seed_sequence)) + seed_sequence
    elif len(seed_sequence) > sequence_length:
        seed_sequence = seed_sequence[-sequence_length:]
    return np.array(seed_sequence)

def start_training_and_generation():
    midi_directory = midi_directory_entry.get()
    user_prompt = prompt_entry.get()
    if not midi_directory or not user_prompt:
        messagebox.showerror("Input Error", "Please provide both the MIDI directory and the musical prompt.")
        return
    
    try:
        status_label.config(text="Processing MIDI files...")
        notes = preprocess_midi_files(midi_directory)
        sequence_length = 100
        (network_input, network_output), pitchnames, note_to_int = create_sequences(notes, sequence_length)
        n_vocab = len(set(notes))
        model = create_model(network_input, n_vocab)
        
        status_label.config(text="Training the model...")
        model.fit(network_input, network_output, epochs=50, batch_size=64)
        
        status_label.config(text="Generating music...")
        seed_sequence = prompt_to_seed(user_prompt, note_to_int, sequence_length)
        prediction_output = generate_notes(model, network_input, pitchnames, n_vocab, seed_sequence, num_notes=500)
        
        file_name = f"ai_generated_music_{user_prompt.replace(' ', '_')}.mid"
        create_midi(prediction_output, file_name)
        
        status_label.config(text=f"Music generation completed. Saved as {file_name}")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {str(e)}")
        status_label.config(text="")

# GUI Implementation
root = tk.Tk()
root.title("AI-Powered Music Generation")

# Labels
ttk.Label(root, text="MIDI Directory:").grid(row=0, column=0, padx=10, pady=10)
ttk.Label(root, text="Musical Prompt:").grid(row=1, column=0, padx=10, pady=10)

# Entries
midi_directory_entry = ttk.Entry(root, width=50)
midi_directory_entry.grid(row=0, column=1, padx=10, pady=10)

prompt_entry = ttk.Entry(root, width=50)
prompt_entry.grid(row=1, column=1, padx=10, pady=10)

# Buttons
ttk.Button(root, text="Browse", command=lambda: midi_directory_entry.insert(0, filedialog.askdirectory())).grid(row=0, column=2, padx=10, pady=10)
ttk.Button(root, text="Start", command=start_training_and_generation).grid(row=2, column=1, padx=10, pady=10)

# Status Label
status_label = ttk.Label(root, text="")
status_label.grid(row=3, column=0, columnspan=3, pady=20)

root.mainloop()