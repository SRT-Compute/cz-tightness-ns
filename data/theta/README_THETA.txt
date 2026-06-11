Place rigidity_v5_events.csv (from the local analysis machine) in this folder.

Columns:
  Theta        blockwise coherence |sum_Q I_Q| / sum_Q |I_Q|  (manuscript Table theta)
  A            axisymmetry index of the local field (auxiliary diagnostic)
  ommax        |omega| at the event
  nblocks      number of admitted blocks (>= 4 points each)
  Th_null_med  median Theta under the within-event random-block-sign null
  pnull        fraction of random-sign draws with Theta >= observed
               (basis of the within-event sign test quoted in the manuscript)
  sigma_loc    contracted strain at the event (auxiliary diagnostic)
  file         source field
  ev           event index
  kind         'real' or 'surrogat' (spectrum-matched phase-randomised surrogate)

Manuscript usage: Table theta = median Theta by (file, kind) with rank-sum p;
the within-event sign test uses pnull on kind='real' rows.
