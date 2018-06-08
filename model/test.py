import pygame,time
bgm_path = r'E:\GAN\NARUTO_game\audio/BGM.mp3'
sound_path1 = r'E:\GAN\NARUTO_game\audio/test1.wav'
sound_path2 = r'E:\GAN\NARUTO_game\audio/test2.wav'
pygame.mixer.init()

pygame.mixer.music.load(bgm_path)
pygame.mixer.music.set_volume(0.2) 
pygame.mixer.music.play()

time.sleep(5)

sound1 = pygame.mixer.Sound(sound_path1)
sound1.set_volume(0.2)
sound1.play()
time.sleep(5)
sound2 = pygame.mixer.Sound(sound_path2)
sound2.set_volume(0.2)
sound2.play()

time.sleep(100)
pygame.mixer.music.stop()


