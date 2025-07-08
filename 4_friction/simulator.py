# Mass-Spring Solids Simulation

import numpy as np  # numpy for linear algebra
import pygame  # pygame for visualization

pygame.init()
import typer

import square_mesh  # square mesh
import time_integrator
import my_time_integrator
from potential.potential_args import PotentialArgs

app = typer.Typer(pretty_exceptions_enable=False)


@app.command()
def main(
    # simulation setup
    side_len: float = 1,
    rho: float = 1000,  # density of square
    k: float = 2e4,  # spring stiffness
    n_seg: int = 4,  # num of segments per side of the square
    h: float = 0.01,  # time step size in s
    mu: float = 0.11,  # friction coefficient of the slope
    use_my_code: bool = False,
):
    DBC = []  # no nodes need to be fixed
    ground_n = np.array([0.1, 1.0])  # normal of the slope
    ground_n /= np.linalg.norm(ground_n)
    ground_o = np.array([0.0, -1.0])  # a point on the slope

    [x, e] = square_mesh.generate(side_len, n_seg)
    v = np.array([[0.0, 0.0]] * len(x))  # velocity
    m = [rho * side_len * side_len / ((n_seg + 1) * (n_seg + 1))] * len(x)
    l2 = [(x[i0] - x[i1]).dot(x[i0] - x[i1]) for i0, i1 in e]
    k = [k] * len(e)
    is_DBC = [False] * len(x)
    for i in DBC:
        is_DBC[i] = True
    contact_area = [side_len / n_seg] * len(x)

    resolution = np.array([900, 900])
    offset = resolution / 2
    scale = 200

    def screen_projection(x):
        return [offset[0] + scale * x[0], resolution[1] - (offset[1] + scale * x[1])]

    time_step = 0
    square_mesh.write_to_file(time_step, x, n_seg)
    screen = pygame.display.set_mode(resolution)
    font = pygame.font.SysFont("Arial", 24)  # <- initialize font
    running = True
    paused = False  # <-- added pause state

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                paused = not paused  # toggle pause state

        print("### Time step", time_step, "###")

        screen.fill((255, 255, 255))

        # Draw time step on screen
        time_text = font.render(f"Time Step: {time_step}", True, (0, 0, 0))
        screen.blit(time_text, (10, 10))

        # Draw slope
        pygame.draw.aaline(
            screen,
            (0, 0, 255),
            screen_projection(
                [ground_o[0] - 3.0 * ground_n[1], ground_o[1] + 3.0 * ground_n[0]]
            ),
            screen_projection(
                [ground_o[0] + 3.0 * ground_n[1], ground_o[1] - 3.0 * ground_n[0]]
            ),
        )

        # Draw springs
        for eI in e:
            pygame.draw.aaline(
                screen,
                (0, 0, 255),
                screen_projection(x[eI[0]]),
                screen_projection(x[eI[1]]),
            )

        # Draw particles
        for xI in x:
            pygame.draw.circle(
                screen,
                (0, 0, 255),
                screen_projection(xI),
                0.1 * side_len / n_seg * scale,
            )

        # Draw velocity vectors
        arrow_scale = 0.1
        for i in range(len(x)):
            start = np.array(screen_projection(x[i]))
            end = np.array(screen_projection(x[i] + arrow_scale * v[i]))
            pygame.draw.line(screen, (255, 0, 0), start, end, 2)

            direction = end - start
            length = np.linalg.norm(direction)
            if length > 1e-8:
                direction /= length
                perp = np.array([-direction[1], direction[0]])
                arrow_size = 5
                tip = end
                left = tip - arrow_size * direction + arrow_size * 0.5 * perp
                right = tip - arrow_size * direction - arrow_size * 0.5 * perp
                pygame.draw.polygon(screen, (255, 0, 0), [tip, left, right])

        pygame.display.flip()

        if not paused:
            if use_my_code:
                [x, v] = my_time_integrator.step_forward(
                    x,
                    e,
                    v,
                    m,
                    l2,
                    k,
                    ground_n,
                    ground_o,
                    contact_area,
                    mu,
                    is_DBC,
                    h,
                    1e-2,
                )
            else:
                [x, v] = time_integrator.step_forward(
                    x,
                    e,
                    v,
                    m,
                    l2,
                    k,
                    ground_n,
                    ground_o,
                    contact_area,
                    mu,
                    is_DBC,
                    h,
                    1e-2,
                )

            time_step += 1
            square_mesh.write_to_file(time_step, x, n_seg)

        pygame.time.wait(int(h * 1000))

    pygame.quit()


if __name__ == "__main__":
    app()
