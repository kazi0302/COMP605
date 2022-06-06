!--------------------------------------------------------------------------------------------------

!~-------------------Hirokatsu Suzuki--------------------------------------------------------------
!-----------------HW3 (MPI 2D Decomposition)-------------------------------------------------------

!--Jacobi program----------------------------------------------------------------------------------

program Jacobi

!--------------------------------------------------------------------------------------------------
    
        implicit none
        include 'mpif.h'
    
!----Claim variables-------------------------------------------------------------------------------
    
        integer :: rank, value, size, errcnt, toterr, i, j, itcnt, i_first, i_last, ie
        integer :: status(MPI_STATUS_SIZE)
        double precision :: diffnorm, gdiffnorm
        double precision, dimension(7, 7) :: xlocal
        double precision, dimension(7,7) :: xnew
        double precision :: t1, t2, diff
    
!-----initialize MPI and start timing--------------------------------------------------------------
    
        call MPI_Init(ie);
    
        t1 = MPI_Wtime()
    
        call MPI_COMM_RANK( MPI_COMM_WORLD, rank, ie)
        call MPI_COMM_SIZE( MPI_COMM_WORLD, size, ie)
    
!---end MPI if processors less than 4-------------------------------------------------------------    
    
        if (size /= 4) then
            call MPI_Abort(MPI_COMM_WORLD, 1)
        end if
    
!---fill in the matrices (initial condition)------------------------------------------------------
    
        do i = 1, 7
            do j = 1, 7
                xlocal(i,j) = rank
            end do
        end do
    
        do j = 1, 7
            xlocal(1,j) = -1
            xlocal(7,j) = -1 
        end do
    
!---main program (Jacobi)-------------------------------------------------------------------------
    
        itcnt = 0
        gdiffnorm = 10.0
    
!---begin do while--------------------------------------------------------------------------------
    
        do while (gdiffnorm > 1.0e-2 .and. itcnt < 100)
    
!---exchange rows---------------------------------------------------------------------------------
    
            if (rank == 0 .or. rank == 1) then
                call MPI_Send( xlocal(6,:), 7, MPI_DOUBLE, rank + 2, 0, MPI_COMM_WORLD, ie )
            end if
    
            if (rank == 2 .or. rank == 3) then
                call MPI_Recv( xlocal(1,:), 7, MPI_DOUBLE, rank - 2, 0, MPI_COMM_WORLD, status, ie )
            end if
    
            if (rank == 2 .or. rank == 3) then
                call MPI_Send( xlocal(2,:), 7, MPI_DOUBLE, rank - 2, 1, MPI_COMM_WORLD, ie)
            end if
            if (rank == 0 .or. rank == 1) then
                call MPI_Recv( xlocal(7,:), 7, MPI_DOUBLE, rank + 2, 1, MPI_COMM_WORLD, status, ie )
            end if
    
!---exchange columns------------------------------------------------------------------------------
    
            if (rank == 0 .or. rank == 2) then
                call MPI_Send( xlocal(:, 6), 7, MPI_DOUBLE, rank + 1, 2, MPI_COMM_WORLD, ie )
            end if
            if (rank == 1 .or. rank == 3) then
                call MPI_Recv( xlocal(:, 1), 7, MPI_DOUBLE, rank - 1, 2, MPI_COMM_WORLD, status, ie )
            end if
    
            if (rank == 1 .or. rank == 3) then
                call MPI_Send( xlocal(:,2), 7, MPI_DOUBLE, rank - 1, 3, MPI_COMM_WORLD, ie )
            end if
            if (rank == 0 .or. rank == 2) then
                call MPI_Recv( xlocal(:,7), 7, MPI_DOUBLE, rank + 1, 3, MPI_COMM_WORLD, status, ie )
            end if
    
!---2D decomposition------------------------------------------------------------------------------
    
            itcnt = itcnt + 1
            diffnorm = 0.0
            do i=2, 6	        
                do j=2, 6 
                    xnew(i,j) = (xlocal(i,j+1) + xlocal(i,j-1) + xlocal(i+1,j) + xlocal(i-1,j)) / 4.0
                    diffnorm = diffnorm + (xnew(i,j) - xlocal(i,j)) * (xnew(i,j) - xlocal(i,j))
                end do
            end do
    
            do i=2, 6 
                do j=2, 6 
                    xlocal(i,j) = xnew(i,j)
                end do
            end do
    
!---combines values from all processes------------------------------------------------------------
    
            call MPI_Allreduce( diffnorm, gdiffnorm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, ie )
        
!---print each iteration and diff-----------------------------------------------------------------
    
            gdiffnorm = sqrt( gdiffnorm )
            if (rank == 0) then
                print '("At iteration",I3, " diff is", ES13.6E2)', itcnt, gdiffnorm
            end if
    
        end do
    
!------end do while-------------------------------------------------------------------------------
    
!---calculate the time for each processor---------------------------------------------------------
    
        t2 = MPI_Wtime()
        diff = t2 - t1
        print *, diff
    
!---end MPI---------------------------------------------------------------------------------------
    
        call MPI_Finalize(ie)
    
!-------------------------------------------------------------------------------------------------
    
end
    
!--end program------------------------------------------------------------------------------------