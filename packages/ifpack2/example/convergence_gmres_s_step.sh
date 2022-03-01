matrixFolder="/home/vault/unrz/unrz002h/matrix/MatrixPower"
matrices="af_shell10.mtx Anderson-16.5.mtx audikw_1.mtx cfd2.mtx crankseg_1.mtx dielFilterV3real.mtx Emilia_923.mtx F1.mtx Fault_639.mtx Flan_1565.mtx G3_circuit.mtx Geo_1438.mtx"
#threads="1 5 10 15 20" #no variation with threads as randomness disables
threads="20"
tmpFolder="tmp"
#solvers="TPETRA@GMRES TPETRA@GMRES@S-STEP"
solvers="TPETRA@GMRES@S-STEP"
precons="NONE:NONE RELAXATION:Jacobi RELAXATION:MT@Gauss-Seidel"
#RELAXATION:Jacobi RELAXATION:MT@Gauss-Seidel RELAXATION:MT@Symmetric@Gauss-Seidel RELAXATION:Gauss-Seidel RELAXATION:Symmetric@Gauss-Seidel"
resultFolder="gmresSstep_results"
rawFolder="${resultFolder}/raw"
mkdir -p ${rawFolder}
#totalIter=2000
#iterStride=100
#iter="10,50,100,150,200,250,300,350,400,450,500,600,700,800,900,1000,1200,1400,1600,1800,2000"
iter="10,50,100,150,200,250,300,350,400,450,500,600,700,800,900,1000,1200,1400,1600,1800,2000,2400,2800,3200,4000"

./check-state.sh config.txt
#get machine env
./machine-state.sh > ${resultFolder}/machine-state.txt

#get matrix names
#cd $matrixFolder
#matrices=
#while read -r i
#do
#    matrices=$matrices" "$i
#done < <(find *)
#echo $matrices
#cd -

mkdir -p ${tmpFolder}
tmpFile=${tmpFolder}/tmp.txt
tmpFile2=${tmpFolder}/tmp2.txt
tmpFile3=${tmpFolder}/tmp3.txt
tmpFile4=${tmpFolder}/tmp4.txt

function pasteToFile_init
{
    file="$1"
    header="$2"
    str="$3"

    echo "${header} ${str}" | tr " " "\n" > ${tmpFile2}
    cat ${file} | head -n 1 > ${tmpFile3} #copy header
    cat ${tmpFile2} >> ${tmpFile3} #copy contents
    mv ${tmpFile3} ${file}
}


function pasteToFile
{
    file="$1"
    header="$2"
    str="$3"

    echo "${header} ${str}" | tr " " "\n" > ${tmpFile2}
    cat ${file} | tail -n +2 > ${tmpFile3}
    paste -d "," ${tmpFile3} ${tmpFile2} > ${tmpFile4}
    cat ${file} | head -n 1 > ${tmpFile2} #copy header
    cat ${tmpFile4} >> ${tmpFile2} #copy contents
    mv ${tmpFile2} ${file}
}

resultAllFile="${resultFolder}/all.txt"
rm -rf ${resultAllFile}

for matrix in ${matrices}; do
    matBasename=$(basename ${matrix} ".mtx")
    rawFile="${rawFolder}/${matBasename}.txt"
    resultFile="${resultFolder}/${matBasename}.txt"

    #printf "%25s, %5s, %5s, %20s, %50s, %20s, %20s, %20s, %20s, %20s\n" "# Matrix" "Thread" "Iter" "Solver" "Precon" "xNorm" "resNorm" "resNorm/bNorm" "errNorm" "Time (s)" > ${resultFile}
    for solver_encryptedName in ${solvers}; do
        solver=$(echo ${solver_encryptedName} | sed "s/@/ /g")
        for precon_encryptedName in ${precons}; do
            precon=$(echo ${precon_encryptedName} | sed "s/@/ /g")
            preconType=$(echo ${precon} | cut -d":" -f1)
            preconSubType=$(echo ${precon} | cut -d":" -f2)
            for thread in ${threads}; do
                #for (( iter=0; iter<${totalIter}; iter=${iter}+${iterStride})); do

                OMP_SCHEDULE=static OMP_PROC_BIND=close OMP_PLACES=cores \
                    OMP_NUM_THREADS=$thread taskset -c 0-$((thread-1)) ./Ifpack2_RelaxationWithEquilibration.exe \
                    --kokkos-numa=1 --kokkos-threads=${thread} \
                    --matrixFilename="${matrixFolder}/${matrix}" \
                    --maxIters=${iter} --convergenceTolerances=1e-10 \
                    --solverTypes="${solver}" \
                    --preconditionerTypes="${preconType}" --preconditionerSubType="${preconSubType}"> ${tmpFile}

                echo "Matrix=${matrix}, Threads = ${thread}, Precon type = ${preconType}, Precon subtype = ${preconSubType}" > ${resultFile}

                cat ${tmpFile} >> ${rawFile}

                iter_string=$(echo ${iter} | tr "," "\n")
                pasteToFile_init ${resultFile} "Iter" "${iter_string}"
                x_norm_w_space=$(grep "||X||_2" ${tmpFile} | cut -d ":" -f 2)
                x_norm=$(echo ${x_norm_w_space})
                pasteToFile ${resultFile} "xNorm" "${x_norm}"
                res_w_space=$(grep "||B-A\*X||_2:" ${tmpFile} | cut -d ":" -f 2)
                res=$(echo ${res_w_space})
                pasteToFile ${resultFile} "resNorm" "${res}"
                res_b_norm_w_space=$(grep "||B-A\*X||_2 / ||B||_2" ${tmpFile} | cut -d ":" -f 2)
                res_b_norm=$(echo ${res_b_norm_w_space})
                pasteToFile ${resultFile} "resNorm/bNorm" "${res_b_norm}"
                #                solveTime_w_space=$(grep "Final solve time " ${tmpFile} | sed 's/.*time //' | cut -d"(" -f1)
                solveTime_w_space=$(grep "solve time (in sec):" ${tmpFile} | cut -d":" -f 2)
                solveTime=$(echo ${solveTime_w_space})
                pasteToFile ${resultFile} "time" "${solveTime}"
                err_norm_w_space=$(grep "||X-X_soln||_" ${tmpFile} | cut -d":" -f 2)
                err_norm=$(echo ${err_norm_w_space})
                pasteToFile ${resultFile} "errNorm" "${err_norm}"
                isConverged_w_space=$(grep "Converged:" ${tmpFile} | cut -d ":" -f 2)
                isConverged=$(echo ${isConverged_w_space})
                pasteToFile ${resultFile} "converged" "${isConverged}"
                numIter_w_space=$(grep "Number of iterations:" ${tmpFile} | cut -d ":" -f 2)
                numIter=$(echo ${numIter_w_space})
                pasteToFile ${resultFile} "Iter" "${numIter}"


                cat ${resultFile} >> ${resultAllFile}
           #     printf "%25s, %5d, %5d, %20s, %50s, %20.12f, %20.12f, %20.12f, %20.12f, %20.12f\n" ${matBasename} ${thread} ${iter} "${solver}" "${precon}" ${x_norm} ${res} ${res_b_norm} ${err_norm} ${solveTime} >> ${resultFile}
                #done
            done
        done
    done
done

rm -rf ${tmpFolder}
cd ${resultFolder}
tar -czvf raw.tar.gz raw
rm -rf raw
cd -
