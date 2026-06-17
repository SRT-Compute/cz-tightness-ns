Columns of rigidity_v5_events.csv (manuscript Table theta):
  Theta        blockwise coherence |sum_Q I_Q| / sum_Q |I_Q|
  A            axisymmetry index of the local field (auxiliary diagnostic)
  ommax        |omega| at the event
  nblocks      number of admitted blocks (>= 4 points each)
  Th_null_med  median Theta under the within-event random-block-sign null
  pnull        fraction of random-sign draws with Theta >= observed
               (basis of the within-event sign test)
  sigma_loc    contracted strain at the event (auxiliary diagnostic)
  file         source field
  ev           event index
  kind         'real' or 'surrogat' (spectrum-matched phase-randomised surrogate)

Table theta = median Theta by (file, kind) with rank-sum p;
the within-event sign test uses pnull on kind='real' rows.
