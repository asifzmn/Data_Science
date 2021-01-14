import numpy as np
from pydub import AudioSegment
from pydub.playback import play

if __name__ == '__main__':

    loop = AudioSegment.from_mp3("/media/az/Misc/Music/Nightcore - Irresistible.mp3")
    # Repeat 2 times
    loop2 = loop * 2
    # Get length in milliseconds
    length = len(loop2)
    # Set fade time
    fade_time = int(length * .5)
    # Fade in and out
    faded = loop2.fade_in(fade_time).fade_out(fade_time)

    # play(loop2)
    soundArray = np.array(loop.get_array_of_samples())
    print(soundArray.shape)
