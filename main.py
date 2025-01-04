import pygame
from car import Car
from track import Track, Drawer

def main():# Initialize Pygame
    pygame.init()
    screen = pygame.display.set_mode((800, 600))  # Set screen size

    # Create track and car instances
    track = Track()  # Ensure Config is set up properly for screen dimensions
    car = Car(track)  # Place car at track's starting point
    drawer = Drawer(screen)

    # Game loop
    running = True
    clock = pygame.time.Clock()

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Handle user input
        keys = pygame.key.get_pressed()
        car.acceleration = keys[pygame.K_UP] - keys[pygame.K_DOWN]
        car.steering = keys[pygame.K_LEFT] - keys[pygame.K_RIGHT]

        # Update car and screen
        car.move()
        # sensor_data = car.get_sensors()
        car.draw(screen)
    
        # screen.fill((0, 0, 0))  # Clear screen
        # drawer.draw_track((255, 255, 255), track.final_points)  # Draw track
        # pygame.draw.rect(screen, (0, 255, 0), (car.x, car.y, car.CAR_WIDTH, car.CAR_LENGTH))  # Draw car
        # pygame.display.flip()

        # Cap the frame rate
        clock.tick(60)

    pygame.quit()

if __name__ == '__main__':
    main()
