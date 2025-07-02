# Mass-Spring Solids Simulation

import numpy as np
import pygame

pygame.init()
import typer
import square_mesh
import time_integrator
from my_time_integrator import step_forward

app = typer.Typer(pretty_exceptions_enable=False)


def run_simulation(
    side_len: float = 1.0,
    rho: float = 1000.0,
    spring_k: float = 2e4,
    n_seg: int = 4,
    time_step_size: float = 0.01,
    y_ground: float = -1.0,
    use_my_code: bool = False,
):

    # simulation setup
    h = time_step_size
    DBC = []  # no nodes need to be fixed

    # initialize simulation
    [x, e] = square_mesh.generate(
        side_len, n_seg
    )  # node positions and edge node indices
    v = np.array([[0.0, 0.0]] * len(x))  # velocity
    m = [rho * side_len * side_len / ((n_seg + 1) * (n_seg + 1))] * len(x)

    # rest length squared
    l2 = []
    for i in range(0, len(e)):
        diff = x[e[i][0]] - x[e[i][1]]
        l2.append(diff.dot(diff))
    k = [spring_k] * len(e)  # spring stiffness

    # identify whether a node is Dirichlet
    is_DBC = [False] * len(x)
    for i in DBC:
        is_DBC[i] = True

    contact_area = [side_len / n_seg] * len(x)  # perimeter split to each node

    # simulation with visualization
    resolution = np.array([900, 900])
    offset = resolution / 2
    scale = 200

    def screen_projection(x):
        return [offset[0] + scale * x[0], resolution[1] - (offset[1] + scale * x[1])]

    time_step = 0
    square_mesh.write_to_file(time_step, x, n_seg)
    screen = pygame.display.set_mode(resolution)
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        print("### Time step", time_step, "###")

        # fill the background and draw the square
        screen.fill((255, 255, 255))
        pygame.draw.aaline(
            screen,
            (0, 0, 255),
            screen_projection([-2, y_ground]),
            screen_projection([2, y_ground]),
        )  # ground

        for eI in e:
            pygame.draw.aaline(
                screen,
                (0, 0, 255),
                screen_projection(x[eI[0]]),
                screen_projection(x[eI[1]]),
            )
        for xI in x:
            pygame.draw.circle(
                screen,
                (0, 0, 255),
                screen_projection(xI),
                0.1 * side_len / n_seg * scale,
            )

        pygame.display.flip()

        # step forward simulation
        if use_my_code:
            [x, v] = step_forward(
                x, e, v, m, l2, k, y_ground, contact_area, is_DBC, h, 1e-2
            )
        else:
            [x, v] = time_integrator.step_forward(
                x, e, v, m, l2, k, y_ground, contact_area, is_DBC, h, 1e-2
            )

        time_step += 1
        pygame.time.wait(int(h * 1000))
        square_mesh.write_to_file(time_step, x, n_seg)

    pygame.quit()


@app.command()
def main(
    side_len: float = typer.Option(1.0, help="Side length of the square"),
    rho: float = typer.Option(1000.0, help="Density of the square"),
    spring_k: float = typer.Option(2e4, help="Spring stiffness"),
    n_seg: int = typer.Option(4, help="Number of segments per side of the square"),
    time_step_size: float = typer.Option(0.01, help="Time step size in seconds"),
    y_ground: float = typer.Option(-1.0, help="Height of the planar ground"),
    use_my_code: bool = typer.Option(
        False, help="Use custom implementation of step_forward"
    ),
):
    """
    Mass-Spring Solids Simulation using either the default or custom implementation.
    """
    run_simulation(
        side_len=side_len,
        rho=rho,
        spring_k=spring_k,
        n_seg=n_seg,
        time_step_size=time_step_size,
        y_ground=y_ground,
        use_my_code=use_my_code,
    )


if __name__ == "__main__":
    app()
