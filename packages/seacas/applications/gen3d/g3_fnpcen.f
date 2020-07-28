C Copyright(C) 1999-2020 National Technology & Engineering Solutions
C of Sandia, LLC (NTESS).  Under the terms of Contract DE-NA0003525 with
C NTESS, the U.S. Government retains certain rights in this software.
C
C See packages/seacas/LICENSE for details

C=======================================================================
      SUBROUTINE FNPCEN (LINK, N4, NUMCOL, NEROW, IELCEN,
     &   IRBOT, IRTOP, NUMROW, NPCEN, NREL, IELCOL, NRNP, NRBOT, NRTOP)
C=======================================================================

C   --*** FNPCEN *** (GEN3D) Compute center block row x column nodes
C   --   Written by Amy Gilkey - revised 04/26/88
C   --   Modified by Greg Sjaardema, 07/06/90
C   --       Problem with noncontiguous material blocks in center
C   --       blocks. (ie. Slidelines separating materials.)
C   --       Problem not fixed, error checking enabled.
C   --       Problem is in setting node numbers in NPCEN.
C   --
C   --FNPCEN sets the number of elements generated by each element (if the
C   --column is within NUMCOL), and returns the nodes by row and column.
C   --
C   --Parameters:
C   --   LINK - IN - the connectivity for the 2D elements, always 4 nodes
C   --   N4 - IN - the number of quadrants
C   --   NUMCOL - IN - the number of columns in the center blocks
C   --   NEROW - IN - the number of element rows in the center blocks
C   --   IELCEN - IN - the element numbers of the center elements
C   --      by column and row
C   --   IRBOT, IRTOP - IN - the row number of the row on top and bottom
C   --   NUMROW - OUT - the number of nodal rows in the center blocks
C   --   NPCEN - OUT - the node numbers of the center nodes by column and row
C   --   NREL - IN/OUT - the number of new elements generated for each element
C   --   IELCOL - IN/OUT - the row number for each element
C   --   NRNP - IN/OUT - the number of new nodes generated for each node
C   --   NRBOT, NRTOP - SCRATCH - size = NEROW
C   --
C   --Common Variables:
C   --   Uses IX1, IX2, IX3, IX4 of /CENPAR/

      INCLUDE 'g3_cenpar.blk'

      INTEGER LINK(4,*)
      INTEGER NPCEN(NUMCOL,*)
      INTEGER IELCEN(NUMCOL,*)
      INTEGER IRBOT(*), IRTOP(*)
      INTEGER NREL(*), IELCOL(*)
      INTEGER NRNP(*)
      INTEGER NRBOT(*), NRTOP(*)

C   --The number of elements generated for center elements is dependent on
C   --the column

      DO 20 ICOL = 1, MAX (NUMCOL-1, 1)
         NR = (ICOL*2-1) * N4
         DO 10 IROW = 1, NEROW
            IEL = IELCEN(ICOL,IROW)
            IF (IEL .LE. 0) GOTO 10
            NREL(IEL) = NR
            IELCOL(IEL) = ICOL
   10    CONTINUE
   20 CONTINUE

C   --Fill in the node rows, checking if bottom or top row has already been
C   --inserted

      CALL INIINT (NEROW, 0, NRBOT)
      CALL INIINT (NEROW, 0, NRTOP)

      NUMROW = 0
      DO 40 IROW = 1, NEROW
         IBOT = NRBOT(IROW)
         IF (IBOT .EQ. 0) THEN
            NUMROW = NUMROW + 1
            IBOT = NUMROW
            IF (IRBOT(IROW) .GT. 0) NRTOP(IRBOT(IROW)) = IBOT
         END IF
         ITOP = NRTOP(IROW)
         IF (ITOP .EQ. 0) THEN
            NUMROW = NUMROW + 1
            ITOP = NUMROW
            IF (IRTOP(IROW) .GT. 0) NRBOT(IRTOP(IROW)) = ITOP
         END IF

         DO 30 ICOL = 1, NUMCOL
            IEL = IELCEN(ICOL,IROW)
            IF (IEL .LE. 0) GOTO 30
            INP = LINK(IX1,IEL)
            IF (NPCEN(ICOL,IBOT) .NE. 0 .AND.
     $          NPCEN(ICOL,IBOT) .NE. INP) THEN
               CALL PRTERR ('FATAL',
     $              'Center blocks must be contiguous')
               CALL PRTERR ('CMDSPEC',
     $              'Slidelines are not allowed')
               STOP 'CENTER BLOCK NONCONTIGUOUS'
            ELSE
               NPCEN(ICOL,IBOT) = INP
            END IF
            INP = LINK(IX4,IEL)
            IF (NPCEN(ICOL,ITOP) .NE. 0 .AND.
     $          NPCEN(ICOL,ITOP) .NE. INP) THEN
               CALL PRTERR ('FATAL',
     $              'Center blocks must be contiguous')
               CALL PRTERR ('CMDSPEC',
     $              'Slidelines are not allowed')
               STOP 'CENTER BLOCK NONCONTIGUOUS'
            ELSE
               NPCEN(ICOL,ITOP) = INP
            END IF
   30    CONTINUE
   40 CONTINUE

C   --The number of nodes generated for center nodes is dependent on
C   --the column

      DO 70 ICOL = 1, MAX (NUMCOL, 1)
         IF (ICOL .EQ. 1) THEN
            NR = 1
         ELSE IF (N4 .EQ. 4) THEN
            NR = (ICOL-1) * 2 * N4
         ELSE
            NR = (ICOL-1) * 2 * N4 + 1
         END IF
         DO 60 IROW = 1, NUMROW
            INP = NPCEN(ICOL,IROW)
            IF (INP .EQ. 0) GOTO 60
            NRNP(INP) = NR
   60    CONTINUE
   70 CONTINUE

      RETURN
      END
