=================================
Using AIM for Experiment Tracking
=================================

Our :class:`~rl_blox.logging.logger.AIMLogger` use
`AIM <https://aimstack.readthedocs.io/en/latest/index.html>`_ to track
experiments.

You can start the AIM UI with

.. code-block:: bash

    aim up

and analyze the results in any browser. An example is displayed below.

.. figure:: _static/aim.png
   :alt: AIM UI
   :align: center
   :width: 100%

   Example of visualization with AIM UI. In this case we tracked 20 runs of
   TD3 with the Pendulum-v1 environment and aggregated the metrics over runs.
