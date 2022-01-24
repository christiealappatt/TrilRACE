//getAllValues() from the matrix
//permute the matrix by passing to RACE
//create a NEW matrix and call setAllValues()
//call expertStaticFillComplete() //see
//muelu/src/Smoothers/MueLu_Amesos2Smoother_def.hpp for example
//give poosibility to return the matrix and permutation vector
//

#ifndef _RACE_PRE_PROCESS_H_
#define _RACE_PRE_PROCESS_H_

#include "Tpetra_CrsMatrix.hpp"
#include <limits.h>
#include <RACE/interface.h>
#include "TrilinosRACE_config.h"
#include "RACE_packtype.hpp"

#define TrilinosRACE_ENCODE_ARG_NUMA(_nrows_, _rowPtr_, _col_, _val_)\
    TrilinosRACE_matValArg<Scalar, LocalOrdinal> *newArg_ = new TrilinosRACE_matValArg<Scalar, LocalOrdinal>;\
    newArg_->rowPtr = _rowPtr_;\
    newArg_->col = _col_;\
    newArg_->val = _val_;\
    void *voidArg = (void*) newArg_;

#define TrilinosRACE_DECODE_ARG_NUMA(_voidArg_)\
    TrilinosRACE_matValArg<Scalar, LocalOrdinal> *nonVoidArg_ = (TrilinosRACE_matValArg<Scalar, LocalOrdinal>*) _voidArg_;\
    size_t* rowPtr = nonVoidArg_->rowPtr;\
    LocalOrdinal* col = nonVoidArg_->col;\
    Scalar* val = nonVoidArg_->val;\

#define TrilinosRACE_DELETE_ARG_NUMA()\
    delete newArg_;

namespace RACE {

    template <class Scalar,class LocalOrdinal>
    struct TrilinosRACE_matValArg
    {
        size_t *rowPtr;
        LocalOrdinal *col;
        Scalar *val;
    };


    //NUMA initializers
    template <class Scalar,class LocalOrdinal>
    void TrilinosRACE_powerInitRowPtrFunc(int start, int end, int pow, int numa_domain, void* arg)
    {
        TrilinosRACE_DECODE_ARG_NUMA(arg);
        if(col != NULL && val != NULL)
        {
            ERROR_PRINT("Something went wrong, I shouldnn' t be here");
        }

        if(pow == 1)
        {
            for(int row=start; row<end; ++row)
            {
                rowPtr[row] = 0*pow;
                rowPtr[row+1] = 0*pow;
            }
        }
        (void) numa_domain;
    }

    template <class Scalar,class LocalOrdinal>
    void TrilinosRACE_powerInitMtxVecFunc(int start, int end, int pow, int numa_domain, void* arg)
    {
        TrilinosRACE_DECODE_ARG_NUMA(arg);
        if(pow==1)
        {
            for(int row=start; row<end; ++row)
            {
                for(int idx=(int)rowPtr[row]; idx<(int)rowPtr[row+1]; ++idx)
                {
                    val[idx] = 0;
                    col[idx] = 0;
                }
            }
        }

        (void) numa_domain;
    }


    //an alternative to if constexpr
    //that is supported even with C++11
    //https://stackoverflow.com/questions/43587405/constexpr-if-alternative
    template <typename T>
//        typename std::enable_if<std::is_same<int, T>::value, void>::type
       void  copyToArrIfNecessary(std::true_type, int *&arr_int, T *arr, int len) {
            arr_int = arr;
        }

    template <typename T>
//        typename std::enable_if<std::is_same<int, T>::value, void>::type
       void copyToArrIfNecessary(std::false_type, int *&arr_int, T *arr, int len) {
#ifdef HAVE_TrilinosRACE_DEBUG
            printf("Creating temporary array\n");
#endif
            arr_int = new int[len];
#pragma omp parallel for schedule(static)
            for(int i=0; i<len; ++i)
            {
                arr_int[i] = arr[i];
            }
        }

    template <typename packtype>
        class preProcess
        {
            private:
            using Scalar = typename packtype::SC;
            using LocalOrdinal = typename packtype::LO;
            using CrsMatrixType = typename packtype::CRS_type;
            //using CrsMatrixType = Tpetra::CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>;
            Teuchos::ParameterList paramList;
            /*Matrix structure*/
            LocalOrdinal nrows;
            int ncols;
            int nnz;
            int *rowPtr_int; //TODO: template this, need to adapt RACE
            int *col_int; //TODO: template this, need to adapt RACE
            const Scalar *val_ptr;

            /*RACE params*/
            double cacheSize;
            int highestPower;
            int* perm;
            int* invPerm;

            /*original matrix*/
            Teuchos::RCP<CrsMatrixType> origA;
            /*permuted matrix*/
            Teuchos::RCP<CrsMatrixType> permA;

            /*Interface to RACE library*/
            RACE::Interface *ce;

            public:
            int* getPerm()
            {
                if(perm == NULL)
                {
                    ERROR_PRINT("Permutation vector not found\n");
                }
                return perm;
            }

            int* getInvPerm()
            {
                if(invPerm == NULL)
                {
                    ERROR_PRINT("Inverse permutation vector not found\n");
                }
                return invPerm;
            }

            Teuchos::RCP<CrsMatrixType> getPermutedMatrix()
            {
                if(permA == Teuchos::null)
                {
                    ERROR_PRINT("Permuted Tpetra::CrsMatrix not found\n");
                }
                return permA;
            }

            RACE::Interface* get_RACE_engine()
            {
                if(!ce)
                {
                    ERROR_PRINT("RACE engine not found\n");
                }
                return ce;
            }

            //constructor
            preProcess(Teuchos::RCP<CrsMatrixType> origA_, Teuchos::ParameterList& paramList): perm(NULL), invPerm(NULL), origA(origA_), permA(Teuchos::null), ce(NULL)
            {
                if(origA == Teuchos::null)
                {
                     ERROR_PRINT("The provided matrix is empty");
                }

#ifdef HAVE_MPI
                int nranks = 1;
                MPI_Comm_size(MPI_COMM_WORLD, &nranks);
                if(nranks != 1)
                {
                    ERROR_PRINT("Currently RACE works only with one MPI process. Only shared memory parallelism is used");
                }
#endif
                //default is a big cache size, so no blocking done by RACE
                cacheSize = std::numeric_limits<double>::max()*1e-9;
                cacheSize = paramList.get("Cache size", std::numeric_limits<double>::max()*1e-9);
                //convert cache size from MB to B
                cacheSize = cacheSize*1000*1000;
                //default highest power is 1, so no temporal blocking
                highestPower = 1;
                highestPower = paramList.get("Highest power", highestPower);

                Teuchos::ArrayRCP<const size_t> rowPointers;
                Teuchos::ArrayRCP<const LocalOrdinal> columnIndices;
                Teuchos::ArrayRCP<const Scalar> values;

                //get matrix arrays
                origA->getAllValues(rowPointers, columnIndices, values);

                //TODO test for int overflow
                //call race on rowPointers and colIndices
                nrows = origA->getNodeNumRows();
                ncols = origA->getNodeNumCols();
                nnz = origA->getNodeNumEntries();

                int eps=100;
                if(nrows > std::numeric_limits<int>::max()-eps  || ncols > std::numeric_limits<int>::max()-eps)
                {
                    ERROR_PRINT("Expect integer overflow, since RACE is not yet prepared to handle large indices");
                }

                //printf("int nrows = %d, ncols = %d\n", nrows, ncols);
                int nthreads = 1; //this parameter not actually used
#pragma omp parallel
                {
                    nthreads = omp_get_num_threads();
                }

                //not ideal to cast const-ness but the RACE_PERMUTE_ON_FLY chek
                //should ensure const-ness is not lost
#ifndef RACE_PERMUTE_ON_FLY
                ERROR_PRINT("Constantness of the matrix will be lost, so please switch on RACE_PERMUTE_ON_FLY, when compiling RACE. Expect errors");
#endif
                //convert rowPointers and column to int*, if they are not
                //Currently RACE can only handle this types
                copyToArrIfNecessary<size_t>(std::integral_constant<bool, std::is_same<int, size_t>::value>{}, rowPtr_int, const_cast<size_t *>(rowPointers.getRawPtr()), nrows+1);
#ifdef HAVE_TrilinosRACE_DEBUG
                printf("RowPtr copy done\n");
#endif
                copyToArrIfNecessary<LocalOrdinal>(std::integral_constant<bool, std::is_same<int, LocalOrdinal>::value>{}, col_int, const_cast<LocalOrdinal *>(columnIndices.getRawPtr()), nnz);
#ifdef HAVE_TrilinosRACE_DEBUG
                printf("Col copy done\n");
#endif

                val_ptr = values.getRawPtr();
/*
                for(int i=0; i<nnz; ++i)
                {
                    int rank=0;
                    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
                    printf("mpirank = %d, col[%d] = %d\n", rank, i, col_int[i]);
                }
*/



#ifdef HAVE_TrilinosRACE_DEBUG
                printf("Starting RACE Pre-Processing\n");
#endif

#ifdef BELOS_TEUCHOS_TIME_MONITOR
                Teuchos::RCP<Teuchos::Time> RACEPreTime;
                RACEPreTime = Teuchos::TimeMonitor::getNewCounter("RACE pre-processing time");
#endif
                {
                    ce = new RACE::Interface(nrows, nthreads, RACE::POWER, rowPtr_int, col_int);
                    int numSharedCache = 1; //currently try only this, and within one socket. Outside socket go MPI
                    ce->RACEColor(highestPower, numSharedCache, cacheSize);
#ifdef BELOS_TEUCHOS_TIME_MONITOR
                    Teuchos::TimeMonitor updateTimer( *RACEPreTime);
#endif
                }

#ifdef HAVE_TrilinosRACE_DEBUG
                printf("Finished RACE Pre-Processing\n");
#endif

               //permute the matrix
                int permLen;
                ce->getPerm(&perm, &permLen);
                ce->getInvPerm(&invPerm, &permLen);
#ifdef HAVE_TrilinosRACE_DEBUG
                printf("Going to generate permuted matrix\n");
#endif
                permuteMatrix(perm, invPerm, true);
               // permuteMatrix(NULL, NULL, true);

                permA->getAllValues(rowPointers, columnIndices, values);
#ifdef HAVE_TrilinosRACE_DEBUG
                printf("Permuted matrix generated\n");
#endif
                //destroy temporarily created arrays
                if(!std::is_same<size_t, int>::value)
                {
                    delete[] rowPtr_int;
                }
                if(!std::is_same<LocalOrdinal, int>::value)
                {
                    delete[] col_int;
                }

            }

            ~preProcess()
            {
                if(ce)
                {
                    delete ce;
                }

                if(perm)
                {
                    delete[] perm;
                }
                if(invPerm)
                {
                    delete[] invPerm;
                }
            }

            //symmetrically permute
            void permuteMatrix(int *_perm_, int*  _invPerm_, bool RACEalloc)
            {
                Scalar* newVal = new Scalar[nnz];
                LocalOrdinal* newCol = new LocalOrdinal[nnz];
                size_t* newRowPtr = new size_t[nrows+1];

                newRowPtr[0] = 0;

#ifdef HAVE_TrilinosRACE_DEBUG
                printf("Going to NUMA init rowPtr\n");
#endif

                if(!RACEalloc)
                {
                    //NUMA init
#pragma omp parallel for schedule(static)
                    for(int row=0; row<nrows; ++row)
                    {
                        newRowPtr[row+1] = 0;
                    }
                }
                else
                {
                    numaInitRowPtr(newRowPtr);
                }

#ifdef HAVE_TrilinosRACE_DEBUG
                printf("NUMA inited rowPtr, will start creating new rowPtr\n");
#endif

                if(_perm_ != NULL)
                {
                    //first find newRowPtr; therefore we can do proper NUMA init
                    int _perm_Idx=0;
                    for(int row=0; row<nrows; ++row)
                    {
                        //row _perm_utation
                        size_t _perm_Row = _perm_[row];
                        for(int idx=rowPtr_int[_perm_Row]; idx<rowPtr_int[_perm_Row+1]; ++idx)
                        {
                            ++_perm_Idx;
                        }
                        newRowPtr[row+1] = _perm_Idx;
                    }
                }
                else
                {
                    for(int row=0; row<nrows+1; ++row)
                    {
                        newRowPtr[row] = rowPtr_int[row];
                    }
                }

#ifdef HAVE_TrilinosRACE_DEBUG
                printf("Created new rowPtr\n");
#endif



                if(RACEalloc)
                {
#ifdef HAVE_TrilinosRACE_DEBUG
                    printf("Going to NUMA init col and val\n");
#endif
                    numaInitMtxVec(newRowPtr, newCol, newVal);
                }
#ifdef HAVE_TrilinosRACE_DEBUG
                printf("NUMA inited col and val, will start creating new col and val\n");
#endif

                bool sortedCol = true;
                //TPETRA expects sorted columns else will have to pass through
                //parameter that it is not sorted
                if(_perm_ != NULL)
                {
                    //with NUMA init
#pragma omp parallel for schedule(static)
                    for(int row=0; row<nrows; ++row)
                    {
                        //row _permutation
                        int _perm_Row = _perm_[row];
                        size_t _perm_Idx; //newRowPtr is size_t
                        int idx;
                        int ctr;

                        int rowLen = newRowPtr[row+1]-newRowPtr[row];
                        std::vector<LocalOrdinal> tmpCol(rowLen);
                        std::vector<Scalar> tmpVal(rowLen);
                        for(_perm_Idx=newRowPtr[row],idx=rowPtr_int[_perm_Row], ctr=0; _perm_Idx<newRowPtr[row+1]; ++idx,++_perm_Idx,++ctr)
                        {
                            tmpCol[ctr] = _invPerm_[col_int[idx]];
                            tmpVal[ctr] = val_ptr[idx];
                            if(!sortedCol)
                            {
                                newCol[_perm_Idx] = tmpCol[ctr];
                                newVal[_perm_Idx] = tmpVal[ctr];
                            }
                        }

                        if(sortedCol)
                        {
                            int rowLen = tmpCol.size();
                            std::vector<LocalOrdinal> perm(rowLen);
                            int ctr=0;
                            for(auto it=perm.begin(); it!=perm.end(); ++it)
                            {
                                *it = ctr;
                                ++ctr;
                            }
                            std::stable_sort(perm.begin(), perm.end(), [&](const int& a, const int& b) {return (tmpCol[a] < tmpCol[b]);} );

                            //now permute col and value
                            std::vector<LocalOrdinal> permCol(rowLen);
                            std::vector<Scalar> permVal(rowLen);
                            for(int i=0; i<rowLen; ++i)
                            {
                                permCol[i] = tmpCol[perm[i]];
                                permVal[i] = tmpVal[perm[i]];
                            }

                            for(_perm_Idx=newRowPtr[row], ctr=0; _perm_Idx<newRowPtr[row+1]; ++_perm_Idx,++ctr)
                            {
                                newCol[_perm_Idx] = permCol[ctr];
                                newVal[_perm_Idx] = permVal[ctr];
                            }
                        }
                    }
                }
                else
                {
#pragma omp parallel for schedule(static)
                    for(int row=0; row<nrows; ++row)
                    {
                        for(size_t idx=newRowPtr[row]; idx<newRowPtr[row+1]; ++idx)
                        {
                            newCol[idx] = col_int[idx];
                            newVal[idx] = val_ptr[idx];
                        }
                    }
                }

#ifdef HAVE_TrilinosRACE_DEBUG
                printf("Created new col and val\n");
                printf("Making ArrayRCP out of pointers\n");
#endif

                //make ArrayRCP out of pointers, pass ownership to ArrayRCP
                Teuchos::ArrayRCP<size_t> newRCP_rowPointers(newRowPtr, 0, nrows+1, true);
                Teuchos::ArrayRCP<LocalOrdinal> newRCP_col(newCol, 0, nnz, true);
                Teuchos::ArrayRCP<Scalar> newRCP_val(newVal, 0, nnz, true);

#ifdef HAVE_TrilinosRACE_DEBUG
                printf("Going to create permuted matrix\n");
#endif

                 using Teuchos::ParameterList;
                Teuchos::RCP<ParameterList> params (new ParameterList ("permCRS"));
                if(sortedCol)
                {
                    params->set("sorted", true);
                }
                else
                {
                    params->set("sorted", false);
                }
                permA = Teuchos::RCP<CrsMatrixType>(new CrsMatrixType(origA->getRowMap(), origA->getColMap(), newRCP_rowPointers, newRCP_col, newRCP_val, params));
#if 0
                using Teuchos::arcp_const_cast;
                permA->setAllValues(arcp_const_cast<size_t>(newRCP_rowPointers), arcp_const_cast<LocalOrdinal>(newRCP_col), arcp_const_cast<Scalar>(newRCP_val));
                //Expect importer==domainMap and exporter==rangeMap. Maybe it
                //might work if its not the case as well, but haven't thought of
                //it.
#endif

#ifdef HAVE_TrilinosRACE_DEBUG
                printf("Calling fill complete\n");
#endif
                permA->expertStaticFillComplete(origA->getDomainMap(), origA->getRangeMap()); //, origA->getCrsGraph()->getImporter(), A->getCrsGraph()->getExporter());
#ifdef HAVE_TrilinosRACE_DEBUG
                printf("Created permuted matrix\n");
#endif


            }

            void numaInitRowPtr(size_t* newRowPtr)
            {
                TrilinosRACE_ENCODE_ARG_NUMA(nrows, newRowPtr, NULL, NULL);
                int fn_id = ce->registerFunction(TrilinosRACE_powerInitRowPtrFunc<Scalar, LocalOrdinal>, voidArg, 1);
                ce->executeFunction(fn_id);
                TrilinosRACE_DELETE_ARG_NUMA();
            }


            void numaInitMtxVec(size_t *newRowPtr, LocalOrdinal *newCol, Scalar *newVal)
            {
                TrilinosRACE_ENCODE_ARG_NUMA(nrows, newRowPtr, newCol, newVal);
                int fn_id = ce->registerFunction(TrilinosRACE_powerInitMtxVecFunc<Scalar, LocalOrdinal>, voidArg, 1);
                ce->executeFunction(fn_id);
                TrilinosRACE_DELETE_ARG_NUMA();
            }


        };

}

#endif
