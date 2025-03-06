import pygame
import sys

class GameWindow:
    def __init__(self, window_size=800):
        # Initialize Pygame first
        pygame.init()
        
        self.window_size = window_size
        self.sidebar_width = 200
        # Initialize with extra width for sidebar
        self.display = pygame.display.set_mode((window_size + self.sidebar_width, window_size))
        pygame.display.set_caption('Mars Space Exploration')
        
        # Colors
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.BLUE = (0, 100, 200)
        self.HOVER_BLUE = (0, 120, 240)
        self.LIGHT_GRAY = (240, 240, 240)
        self.RED = (255, 0, 0)
        
        # Button properties
        self.button_width = 200
        self.button_height = 60
        
        # Font setup
        self.info_font = pygame.font.Font(None, 32)
        self.input_font = pygame.font.Font(None, 28)
        
        # Input field properties
        self.input_height = 30
        self.input_width = 100
        self.label_width = 180
        self.input_spacing = 40
        self.input_start_y = 100
        
        # Default parameter values
        self.parameters = {
            "Grid Size": "5",
            "Max Epsilon": "1.0",
            "Min Epsilon": "0.05",
            "Epsilon Decay": "0.0005",
            "Alpha": "0.7",
            "Gamma": "0.7",
            "Episodes": "2000",
            "Max Steps": "100",
            "Policy": "episilon_greedy",
            "Max Temperature": "100",
            "Min Temperature": "0.1",
            "Temp Decay": "0.005"
        }
        
        self.active_input = None
        self.input_rects = {}
        
    def draw_sidebar(self, episode, step):
        # Draw sidebar background
        sidebar_rect = pygame.Rect(0, 0, self.sidebar_width, self.window_size)
        pygame.draw.rect(self.display, self.LIGHT_GRAY, sidebar_rect)
        pygame.draw.line(self.display, self.BLACK, (self.sidebar_width, 0), 
                        (self.sidebar_width, self.window_size), 2)
        
        # Draw episode info
        episode_text = self.info_font.render(f"Episode: {episode}", True, self.BLACK)
        self.display.blit(episode_text, (20, 30))
        
        # Draw step info
        step_text = self.info_font.render(f"Step: {step}", True, self.BLACK)
        self.display.blit(step_text, (20, 70))

    def draw_input_fields(self):
        y = self.input_start_y
        for label, value in self.parameters.items():
            # Draw label
            label_text = self.input_font.render(label + ":", True, self.BLACK)
            self.display.blit(label_text, (20, y))
            
            # Draw input box
            input_rect = pygame.Rect(self.label_width + 20, y, self.input_width, self.input_height)
            self.input_rects[label] = input_rect
            
            color = self.HOVER_BLUE if self.active_input == label else self.WHITE
            pygame.draw.rect(self.display, color, input_rect)
            pygame.draw.rect(self.display, self.BLACK, input_rect, 1)
            
            # Draw input text
            text = self.input_font.render(value, True, self.BLACK)
            text_rect = text.get_rect(midleft=(input_rect.left + 5, input_rect.centery))
            self.display.blit(text, text_rect)
            
            y += self.input_spacing

    def handle_input_click(self, pos):
        for label, rect in self.input_rects.items():
            if rect.collidepoint(pos):
                self.active_input = label
                return True
        self.active_input = None
        return False

    def handle_key_input(self, event):
        if self.active_input:
            if event.key == pygame.K_RETURN:
                self.active_input = None
            elif event.key == pygame.K_BACKSPACE:
                self.parameters[self.active_input] = self.parameters[self.active_input][:-1]
            else:
                current = self.parameters[self.active_input]
                if len(current) < 10:  # Limit input length
                    self.parameters[self.active_input] += event.unicode

    def validate_parameters(self):
        try:
            # Convert and validate all numeric parameters
            grid_size = int(self.parameters["Grid Size"])
            max_epsilon = float(self.parameters["Max Epsilon"])
            min_epsilon = float(self.parameters["Min Epsilon"])
            epsilon_decay = float(self.parameters["Epsilon Decay"])
            alpha = float(self.parameters["Alpha"])
            gamma = float(self.parameters["Gamma"])
            episodes = int(self.parameters["Episodes"])
            max_steps = int(self.parameters["Max Steps"])
            max_temp = float(self.parameters["Max Temperature"])
            min_temp = float(self.parameters["Min Temperature"])
            temp_decay = float(self.parameters["Temp Decay"])
            
            # Basic validation
            if not (0 < grid_size <= 10):
                return False, "Grid size must be between 1 and 10"
            if not (0 <= min_epsilon <= max_epsilon <= 1):
                return False, "Invalid epsilon range"
            if not (0 <= alpha <= 1 and 0 <= gamma <= 1):
                return False, "Alpha and Gamma must be between 0 and 1"
            if episodes <= 0 or max_steps <= 0:
                return False, "Episodes and Max Steps must be positive"
            if not (0 <= min_temp <= max_temp):
                return False, "Invalid temperature range"
            
            return True, None
        except ValueError:
            return False, "Invalid numeric input"

    def show_start_screen(self):
        title_font = pygame.font.Font(None, 64)
        button_font = pygame.font.Font(None, 36)
        
        # Adjust title position to account for sidebar
        title_text = title_font.render("Mars Exploration", True, self.BLACK)
        title_rect = title_text.get_rect(center=(self.sidebar_width + self.window_size/2, self.window_size/3))
        
        # Adjust button position to account for sidebar
        button_x = self.sidebar_width + self.window_size/2 - self.button_width/2
        button_y = self.window_size/2
        button_rect = pygame.Rect(button_x, button_y, self.button_width, self.button_height)
        
        start_text = button_font.render("Start Mission", True, self.WHITE)
        text_rect = start_text.get_rect(center=button_rect.center)
        
        pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_ARROW)
        
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                    
                if event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_pos = pygame.mouse.get_pos()
                    if button_rect.collidepoint(mouse_pos):
                        pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_ARROW)
                        return True
            
            self.display.fill(self.WHITE)
            
            # Draw sidebar even in start screen
            sidebar_rect = pygame.Rect(0, 0, self.sidebar_width, self.window_size)
            pygame.draw.rect(self.display, self.LIGHT_GRAY, sidebar_rect)
            pygame.draw.line(self.display, self.BLACK, (self.sidebar_width, 0), 
                           (self.sidebar_width, self.window_size), 2)
            
            self.display.blit(title_text, title_rect)
            
            mouse_pos = pygame.mouse.get_pos()
            if button_rect.collidepoint(mouse_pos):
                pygame.draw.rect(self.display, self.HOVER_BLUE, button_rect, border_radius=10)
                pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_HAND)
            else:
                pygame.draw.rect(self.display, self.BLUE, button_rect, border_radius=10)
                pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_ARROW)
            
            self.display.blit(start_text, text_rect)
            
            pygame.display.flip() 