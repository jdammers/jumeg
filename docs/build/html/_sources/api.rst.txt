.. _api_documentation:

=================
API Documentation
=================

.. currentmodule:: jumeg

Connectivity
============

Functions

.. currentmodule:: jumeg.connectivity

.. autosummary::
   :toctree: generated/
   :nosignatures:

    find_distances_matrix
    weighted_con_matrix
    get_label_distances
    make_annot_from_csv

    sensor_connectivity_3d
    plot_grouped_connectivity_circle
    plot_generic_grouped_circle
    plot_grouped_causality_circle
    plot_degree_circle
    plot_lines_and_blobs
    plot_labelled_group_connectivity_circle
    plot_fica_grouped_circle

.. currentmodule:: jumeg.connectivity.causality

.. autosummary::
   :toctree: generated/
   :nosignatures:

    dw_whiteness
    consistency
    do_mvar_evaluation
    check_whiteness_and_consistency
    check_model_order
    prepare_causality_matrix
    make_frequency_bands
    compute_order_extended
    compute_order
    compute_causal_outflow_inflow
